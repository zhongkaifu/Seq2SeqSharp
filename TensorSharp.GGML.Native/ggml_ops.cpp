#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-cpu.h"

// GGML context memory pool: reuse mem_buffers to avoid per-op allocation overhead
namespace ggml_pool
{
    constexpr std::size_t k_pool_buffer_size = 32 * 1024 * 1024;  // 32 MB, covers larger ops and batched matmul
    constexpr int k_pool_initial_count = 8;
    constexpr int k_pool_max_count = 32;

    struct PoolEntry
    {
        void* ptr = nullptr;
        std::size_t size = 0;
    };

    static std::mutex g_pool_mutex;
    static std::vector<PoolEntry> g_pool;

    static void* pool_alloc(std::size_t size)
    {
        if (size == 0 || size > k_pool_buffer_size)
            return nullptr;
        void* ptr = std::malloc(size);
        return ptr;
    }

    static void pool_free(void* ptr)
    {
        if (ptr != nullptr)
            std::free(ptr);
    }

    static PoolEntry acquire(std::size_t required_size)
    {
        if (required_size == 0 || required_size > k_pool_buffer_size)
            return {};
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        for (auto it = g_pool.begin(); it != g_pool.end(); ++it)
        {
            if (it->size >= required_size)
            {
                PoolEntry e = *it;
                g_pool.erase(it);
                return e;
            }
        }
        void* ptr = pool_alloc(k_pool_buffer_size);
        if (ptr == nullptr)
            return {};
        return { ptr, k_pool_buffer_size };
    }

    static void release(PoolEntry e)
    {
        if (e.ptr == nullptr)
            return;
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        if (g_pool.size() < static_cast<std::size_t>(k_pool_max_count))
            g_pool.push_back(e);
        else
            pool_free(e.ptr);
    }

    static void ensure_initial_pool()
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        while (g_pool.size() < static_cast<std::size_t>(k_pool_initial_count))
        {
            void* ptr = pool_alloc(k_pool_buffer_size);
            if (ptr == nullptr)
                break;
            g_pool.push_back({ ptr, k_pool_buffer_size });
        }
    }
}

#if defined(__clang__) || defined(__GNUC__)
#define TSG_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define TSG_EXPORT extern "C"
#endif

namespace
{
    struct TensorView2DDesc
    {
        void* data;
        int dim0;
        int dim1;
        int stride0;
        int stride1;
        std::int64_t raw_bytes;
    };

    struct TensorView3DDesc
    {
        void* data;
        int dim0;
        int dim1;
        int dim2;
        int stride0;
        int stride1;
        int stride2;
        std::int64_t raw_bytes;
    };

    struct TensorView4DDesc
    {
        void* data;
        int ne0;
        int ne1;
        int ne2;
        int ne3;
        std::int64_t nb1;
        std::int64_t nb2;
        std::int64_t nb3;
        std::int64_t raw_bytes;
    };

    struct ContiguousTensorDesc
    {
        void* data;
        std::int64_t element_count;
    };

    enum class UnaryOpCode : int
    {
        Neg = 1,
        Exp = 2,
        Log = 3,
        Sqrt = 4,
        Relu = 5,
        Sigmoid = 6,
        Tanh = 7,
        SiLU = 8,
        Step = 9,
        Abs = 10,
        Sign = 11,
    };

    enum class BinaryTensorOpCode : int
    {
        Add = 1,
        Sub = 2,
        Mul = 3,
        Div = 4,
    };

    enum class BinaryScalarOpCode : int
    {
        Add = 1,
        Sub = 2,
        ReverseSub = 3,
        Mul = 4,
        Div = 5,
        ReverseDiv = 6,
    };

    enum class ActivationGradOpCode : int
    {
        Relu = 1,
        Sigmoid = 2,
        Tanh = 3,
        SiLU = 4,
    };

    enum class NormOpCode : int
    {
        LayerNorm = 1,
        RmsNorm = 2,
    };

    enum class ReductionOpCode : int
    {
        Sum = 1,
        Mean = 2,
    };

    enum class IndexReductionOpCode : int
    {
        Argmin = 1,
        Argmax = 2,
    };

    struct TensorBinding
    {
        ggml_tensor* storage = nullptr;
        ggml_tensor* tensor = nullptr;
        std::size_t raw_bytes = 0;
    };

    // For zero-copy path: binding + buffer that must stay alive
    struct HostPtrBinding
    {
        TensorBinding binding;
        ggml_backend_buffer_t buffer = nullptr;
    };

    thread_local std::string g_last_error;
    std::once_flag g_backend_init_once;
    ggml_backend_t g_backend = nullptr;
    int g_backend_type = 0;

    void set_last_error(const std::string& message)
    {
        g_last_error = message;
    }

    void clear_last_error()
    {
        g_last_error.clear();
    }

    constexpr int BACKEND_TYPE_METAL = 1;
    constexpr int BACKEND_TYPE_CPU = 2;

    void initialize_backend()
    {
        clear_last_error();

        if (g_backend_type == BACKEND_TYPE_METAL)
        {
            g_backend = ggml_backend_metal_init();
            if (g_backend == nullptr)
            {
                set_last_error("ggml-metal backend initialization failed.");
                return;
            }
        }
        else if (g_backend_type == BACKEND_TYPE_CPU)
        {
            g_backend = ggml_backend_cpu_init();
            if (g_backend == nullptr)
            {
                set_last_error("ggml-cpu backend initialization failed.");
                return;
            }
        }
        else
        {
            set_last_error("Unknown GGML backend type requested.");
            return;
        }

        ggml_pool::ensure_initial_pool();
    }

    bool ensure_backend(int backend_type)
    {
        if (backend_type != BACKEND_TYPE_METAL && backend_type != BACKEND_TYPE_CPU)
        {
            set_last_error("Invalid GGML backend type.");
            return false;
        }

        if (g_backend_type == 0)
        {
            g_backend_type = backend_type;
        }
        else if (g_backend_type != backend_type)
        {
            set_last_error("A different GGML backend was already initialized in this process.");
            return false;
        }

        std::call_once(g_backend_init_once, initialize_backend);
        return g_backend != nullptr;
    }

    bool ensure_backend()
    {
        const int backend_type = (g_backend_type == 0) ? BACKEND_TYPE_METAL : g_backend_type;
        return ensure_backend(backend_type);
    }

    bool backend_supports_op(ggml_tensor* op)
    {
        return op != nullptr && g_backend != nullptr && ggml_backend_supports_op(g_backend, op);
    }

    struct ContextHandle
    {
        ggml_context* value = nullptr;

        explicit ContextHandle(ggml_context* ctx)
            : value(ctx)
        {
        }

        ~ContextHandle()
        {
            if (value != nullptr)
            {
                ggml_free(value);
            }
        }

        ContextHandle(const ContextHandle&) = delete;
        ContextHandle& operator=(const ContextHandle&) = delete;
    };

    // Pooled context: uses memory pool for ggml context buffer, returns buffer to pool on destruction
    struct PooledContextHandle
    {
        ggml_context* value = nullptr;
        ggml_pool::PoolEntry pool_entry;

        PooledContextHandle() = default;

        bool init(std::size_t required_size)
        {
            pool_entry = ggml_pool::acquire(required_size);
            if (pool_entry.ptr == nullptr)
                return false;
            ggml_init_params params = {};
            params.mem_size = pool_entry.size;
            params.mem_buffer = pool_entry.ptr;
            params.no_alloc = true;
            value = ggml_init(params);
            if (value == nullptr)
            {
                ggml_pool::release(pool_entry);
                pool_entry = {};
                return false;
            }
            return true;
        }

        ~PooledContextHandle()
        {
            if (value != nullptr)
            {
                ggml_free(value);
                value = nullptr;
            }
            if (pool_entry.ptr != nullptr)
            {
                ggml_pool::release(pool_entry);
                pool_entry = {};
            }
        }

        PooledContextHandle(const PooledContextHandle&) = delete;
        PooledContextHandle& operator=(const PooledContextHandle&) = delete;
    };

    struct BufferHandle
    {
        ggml_backend_buffer_t value = nullptr;

        explicit BufferHandle(ggml_backend_buffer_t buffer)
            : value(buffer)
        {
        }

        ~BufferHandle()
        {
            if (value != nullptr)
            {
                ggml_backend_buffer_free(value);
            }
        }

        BufferHandle(const BufferHandle&) = delete;
        BufferHandle& operator=(const BufferHandle&) = delete;

        BufferHandle(BufferHandle&& other) noexcept
            : value(other.value)
        {
            other.value = nullptr;
        }

        BufferHandle& operator=(BufferHandle&& other) noexcept
        {
            if (this != &other)
            {
                if (value != nullptr)
                    ggml_backend_buffer_free(value);
                value = other.value;
                other.value = nullptr;
            }
            return *this;
        }
    };

    template <std::size_t N>
    bool is_non_overlapping_fast_to_slow(const std::array<int, N>& sizes, const std::array<int, N>& strides)
    {
        std::int64_t required_stride = 1;
        for (std::size_t i = 0; i < N; ++i)
        {
            if (sizes[i] <= 0 || strides[i] < 0)
            {
                return false;
            }

            if (sizes[i] == 1)
            {
                continue;
            }

            if (strides[i] < required_stride)
            {
                return false;
            }

            required_stride = static_cast<std::int64_t>(strides[i]) * sizes[i];
        }

        return true;
    }

    std::size_t required_raw_bytes(const TensorView2DDesc& desc)
    {
        const std::int64_t max_offset =
            (static_cast<std::int64_t>(desc.dim0) - 1) * desc.stride0 +
            (static_cast<std::int64_t>(desc.dim1) - 1) * desc.stride1;
        return static_cast<std::size_t>((max_offset + 1) * sizeof(float));
    }

    std::size_t required_raw_bytes(const TensorView3DDesc& desc)
    {
        const std::int64_t max_offset =
            (static_cast<std::int64_t>(desc.dim0) - 1) * desc.stride0 +
            (static_cast<std::int64_t>(desc.dim1) - 1) * desc.stride1 +
            (static_cast<std::int64_t>(desc.dim2) - 1) * desc.stride2;
        return static_cast<std::size_t>((max_offset + 1) * sizeof(float));
    }

    std::size_t logical_bytes(const TensorView2DDesc& desc)
    {
        return static_cast<std::size_t>(desc.dim0) * desc.dim1 * sizeof(float);
    }

    std::size_t logical_bytes(const TensorView3DDesc& desc)
    {
        return static_cast<std::size_t>(desc.dim0) * desc.dim1 * desc.dim2 * sizeof(float);
    }

    std::size_t required_raw_bytes(const TensorView4DDesc& desc)
    {
        const std::int64_t max_offset =
            (static_cast<std::int64_t>(desc.ne0) - 1) +
            (static_cast<std::int64_t>(desc.ne1) - 1) * (desc.nb1 / static_cast<std::int64_t>(sizeof(float))) +
            (static_cast<std::int64_t>(desc.ne2) - 1) * (desc.nb2 / static_cast<std::int64_t>(sizeof(float))) +
            (static_cast<std::int64_t>(desc.ne3) - 1) * (desc.nb3 / static_cast<std::int64_t>(sizeof(float)));
        return static_cast<std::size_t>((max_offset + 1) * sizeof(float));
    }

    std::size_t logical_bytes(const TensorView4DDesc& desc)
    {
        return static_cast<std::size_t>(desc.ne0) * desc.ne1 * desc.ne2 * desc.ne3 * sizeof(float);
    }

    bool validate_desc(const TensorView2DDesc& desc, const char* name)
    {
        if (desc.data == nullptr)
        {
            set_last_error(std::string("Null pointer passed for ") + name + '.');
            return false;
        }

        if (desc.dim0 <= 0 || desc.dim1 <= 0)
        {
            set_last_error(std::string("Invalid tensor shape passed for ") + name + '.');
            return false;
        }

        if (desc.stride0 < 0 || desc.stride1 < 0)
        {
            set_last_error(std::string("Negative tensor strides are not supported for ") + name + '.');
            return false;
        }

        if (desc.raw_bytes <= 0 || (desc.raw_bytes % static_cast<std::int64_t>(sizeof(float))) != 0)
        {
            set_last_error(std::string("Invalid raw byte size passed for ") + name + '.');
            return false;
        }

        if (static_cast<std::size_t>(desc.raw_bytes) < required_raw_bytes(desc))
        {
            set_last_error(std::string("Raw byte span is too small for ") + name + '.');
            return false;
        }

        return true;
    }

    bool validate_desc(const TensorView3DDesc& desc, const char* name)
    {
        if (desc.data == nullptr)
        {
            set_last_error(std::string("Null pointer passed for ") + name + '.');
            return false;
        }

        if (desc.dim0 <= 0 || desc.dim1 <= 0 || desc.dim2 <= 0)
        {
            set_last_error(std::string("Invalid tensor shape passed for ") + name + '.');
            return false;
        }

        if (desc.stride0 < 0 || desc.stride1 < 0 || desc.stride2 < 0)
        {
            set_last_error(std::string("Negative tensor strides are not supported for ") + name + '.');
            return false;
        }

        if (desc.raw_bytes <= 0 || (desc.raw_bytes % static_cast<std::int64_t>(sizeof(float))) != 0)
        {
            set_last_error(std::string("Invalid raw byte size passed for ") + name + '.');
            return false;
        }

        if (static_cast<std::size_t>(desc.raw_bytes) < required_raw_bytes(desc))
        {
            set_last_error(std::string("Raw byte span is too small for ") + name + '.');
            return false;
        }

        return true;
    }

    bool validate_desc(const TensorView4DDesc& desc, const char* name)
    {
        if (desc.data == nullptr)
        {
            set_last_error(std::string("Null pointer passed for ") + name + '.');
            return false;
        }

        if (desc.ne0 <= 0 || desc.ne1 <= 0 || desc.ne2 <= 0 || desc.ne3 <= 0)
        {
            set_last_error(std::string("Invalid tensor shape passed for ") + name + '.');
            return false;
        }

        if (desc.nb1 <= 0 || desc.nb2 <= 0 || desc.nb3 <= 0)
        {
            set_last_error(std::string("Invalid tensor strides passed for ") + name + '.');
            return false;
        }

        if ((desc.nb1 % static_cast<std::int64_t>(sizeof(float))) != 0
            || (desc.nb2 % static_cast<std::int64_t>(sizeof(float))) != 0
            || (desc.nb3 % static_cast<std::int64_t>(sizeof(float))) != 0)
        {
            set_last_error(std::string("Tensor byte strides must be multiples of sizeof(float) for ") + name + '.');
            return false;
        }

        if (desc.raw_bytes <= 0 || (desc.raw_bytes % static_cast<std::int64_t>(sizeof(float))) != 0)
        {
            set_last_error(std::string("Invalid raw byte size passed for ") + name + '.');
            return false;
        }

        if (static_cast<std::size_t>(desc.raw_bytes) < required_raw_bytes(desc))
        {
            set_last_error(std::string("Raw byte span is too small for ") + name + '.');
            return false;
        }

        return true;
    }

    bool validate_desc(const ContiguousTensorDesc& desc, const char* name)
    {
        if (desc.data == nullptr)
        {
            set_last_error(std::string("Null pointer passed for ") + name + '.');
            return false;
        }

        if (desc.element_count <= 0)
        {
            set_last_error(std::string("Invalid element count passed for ") + name + '.');
            return false;
        }

        return true;
    }

    bool can_map_standard_view(const TensorView2DDesc& desc)
    {
        return desc.stride1 == 1 &&
            is_non_overlapping_fast_to_slow<2>({ desc.dim1, desc.dim0 }, { desc.stride1, desc.stride0 });
    }

    bool can_map_standard_view(const TensorView3DDesc& desc)
    {
        return desc.stride2 == 1 &&
            is_non_overlapping_fast_to_slow<3>({ desc.dim2, desc.dim1, desc.dim0 }, { desc.stride2, desc.stride1, desc.stride0 });
    }

    bool can_map_standard_view(const TensorView4DDesc& desc)
    {
        const auto stride1 = static_cast<int>(desc.nb1 / static_cast<std::int64_t>(sizeof(float)));
        const auto stride2 = static_cast<int>(desc.nb2 / static_cast<std::int64_t>(sizeof(float)));
        const auto stride3 = static_cast<int>(desc.nb3 / static_cast<std::int64_t>(sizeof(float)));

        return is_non_overlapping_fast_to_slow<4>({ desc.ne0, desc.ne1, desc.ne2, desc.ne3 }, { 1, stride1, stride2, stride3 });
    }

    bool can_map_m2_direct(const TensorView2DDesc& desc)
    {
        return desc.stride0 == 1 &&
            is_non_overlapping_fast_to_slow<2>({ desc.dim0, desc.dim1 }, { desc.stride0, desc.stride1 });
    }

    bool can_map_m2_direct(const TensorView3DDesc& desc)
    {
        return desc.stride1 == 1 &&
            is_non_overlapping_fast_to_slow<3>({ desc.dim1, desc.dim2, desc.dim0 }, { desc.stride1, desc.stride2, desc.stride0 });
    }

    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView2DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_2d(ctx, base, desc.dim1, desc.dim0, static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView3DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_3d(
            ctx,
            base,
            desc.dim2,
            desc.dim1,
            desc.dim0,
            static_cast<std::size_t>(desc.stride1) * sizeof(float),
            static_cast<std::size_t>(desc.stride0) * sizeof(float),
            0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView4DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_4d(
            ctx,
            base,
            desc.ne0,
            desc.ne1,
            desc.ne2,
            desc.ne3,
            static_cast<std::size_t>(desc.nb1),
            static_cast<std::size_t>(desc.nb2),
            static_cast<std::size_t>(desc.nb3),
            0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    TensorBinding create_contiguous_binding(ggml_context* ctx, const ContiguousTensorDesc& desc)
    {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.element_count);
        return { tensor, tensor, static_cast<std::size_t>(desc.element_count * static_cast<std::int64_t>(sizeof(float))) };
    }

    TensorBinding create_direct_m2_binding(ggml_context* ctx, const TensorView2DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_2d(ctx, base, desc.dim0, desc.dim1, static_cast<std::size_t>(desc.stride1) * sizeof(float), 0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    TensorBinding create_direct_m2_binding(ggml_context* ctx, const TensorView3DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_3d(
            ctx,
            base,
            desc.dim1,
            desc.dim2,
            desc.dim0,
            static_cast<std::size_t>(desc.stride2) * sizeof(float),
            static_cast<std::size_t>(desc.stride0) * sizeof(float),
            0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    std::vector<float> pack_m2(const TensorView2DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1);

        for (int row = 0; row < desc.dim0; ++row)
        {
            for (int col = 0; col < desc.dim1; ++col)
            {
                packed[(static_cast<std::size_t>(col) * desc.dim0) + row] =
                    data[(static_cast<std::size_t>(row) * desc.stride0) + (static_cast<std::size_t>(col) * desc.stride1)];
            }
        }

        return packed;
    }

    std::vector<float> pack_m2(const TensorView3DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1 * desc.dim2);

        for (int batch = 0; batch < desc.dim0; ++batch)
        {
            for (int row = 0; row < desc.dim1; ++row)
            {
                for (int col = 0; col < desc.dim2; ++col)
                {
                    packed[((static_cast<std::size_t>(batch) * desc.dim2 + col) * desc.dim1) + row] =
                        data[(static_cast<std::size_t>(batch) * desc.stride0) +
                             (static_cast<std::size_t>(row) * desc.stride1) +
                             (static_cast<std::size_t>(col) * desc.stride2)];
                }
            }
        }

        return packed;
    }

    std::vector<float> pack_standard(const TensorView2DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1);

        for (int row = 0; row < desc.dim0; ++row)
        {
            for (int col = 0; col < desc.dim1; ++col)
            {
                packed[(static_cast<std::size_t>(row) * desc.dim1) + col] =
                    data[(static_cast<std::size_t>(row) * desc.stride0) + (static_cast<std::size_t>(col) * desc.stride1)];
            }
        }

        return packed;
    }

    std::vector<float> pack_standard(const TensorView3DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1 * desc.dim2);

        for (int batch = 0; batch < desc.dim0; ++batch)
        {
            for (int row = 0; row < desc.dim1; ++row)
            {
                for (int col = 0; col < desc.dim2; ++col)
                {
                    packed[((static_cast<std::size_t>(batch) * desc.dim1 + row) * desc.dim2) + col] =
                        data[(static_cast<std::size_t>(batch) * desc.stride0) +
                             (static_cast<std::size_t>(row) * desc.stride1) +
                             (static_cast<std::size_t>(col) * desc.stride2)];
                }
            }
        }

        return packed;
    }

    TensorBinding create_packed_m2_binding(ggml_context* ctx, const TensorView2DDesc& desc, std::vector<float>& packed)
    {
        packed = pack_m2(desc);
        ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, desc.dim0, desc.dim1);
        return { tensor, tensor, packed.size() * sizeof(float) };
    }

    TensorBinding create_packed_m2_binding(ggml_context* ctx, const TensorView3DDesc& desc, std::vector<float>& packed)
    {
        packed = pack_m2(desc);
        ggml_tensor* tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, desc.dim1, desc.dim2, desc.dim0);
        return { tensor, tensor, packed.size() * sizeof(float) };
    }

    TensorBinding create_packed_standard_binding(ggml_context* ctx, const TensorView2DDesc& desc, std::vector<float>& packed)
    {
        packed = pack_standard(desc);
        ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, desc.dim1, desc.dim0);
        return { tensor, tensor, packed.size() * sizeof(float) };
    }

    TensorBinding create_packed_standard_binding(ggml_context* ctx, const TensorView3DDesc& desc, std::vector<float>& packed)
    {
        packed = pack_standard(desc);
        ggml_tensor* tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, desc.dim2, desc.dim1, desc.dim0);
        return { tensor, tensor, packed.size() * sizeof(float) };
    }

    void upload_binding(const TensorBinding& binding, const void* data, std::size_t size)
    {
        ggml_backend_tensor_set(binding.storage, data, 0, size);
    }

    // Create a binding that uses host ptr directly as Metal shared memory (zero host-device copies on Apple Silicon).
    // Returns empty binding on failure. Caller must keep buffer_handle alive until compute completes.
    bool create_binding_from_host_ptr_2d(
        ggml_context* ctx,
        ggml_backend_t backend,
        const TensorView2DDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.buffer_from_host_ptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_2d(ctx, base, desc.dim1, desc.dim0, static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    // Zero-copy for m2 (direct layout: stride0==1, column-major)
    bool create_binding_from_host_ptr_direct_m2_2d(
        ggml_context* ctx,
        ggml_backend_t backend,
        const TensorView2DDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.buffer_from_host_ptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_2d(ctx, base, desc.dim0, desc.dim1, static_cast<std::size_t>(desc.stride1) * sizeof(float), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    bool create_binding_from_host_ptr_3d(
        ggml_context* ctx,
        ggml_backend_t backend,
        const TensorView3DDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.buffer_from_host_ptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_3d(ctx, base, desc.dim2, desc.dim1, desc.dim0,
            static_cast<std::size_t>(desc.stride1) * sizeof(float),
            static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    bool create_binding_from_host_ptr_direct_m2_3d(
        ggml_context* ctx,
        ggml_backend_t backend,
        const TensorView3DDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.buffer_from_host_ptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_3d(ctx, base, desc.dim1, desc.dim2, desc.dim0,
            static_cast<std::size_t>(desc.stride2) * sizeof(float),
            static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    bool create_binding_from_host_ptr_4d(
        ggml_context* ctx,
        ggml_backend_t backend,
        const TensorView4DDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.buffer_from_host_ptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_4d(ctx, base, desc.ne0, desc.ne1, desc.ne2, desc.ne3,
            static_cast<std::size_t>(desc.nb1),
            static_cast<std::size_t>(desc.nb2),
            static_cast<std::size_t>(desc.nb3), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    bool create_binding_from_host_ptr_contiguous(
        ggml_context* ctx,
        ggml_backend_t backend,
        const ContiguousTensorDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.buffer_from_host_ptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.element_count) * sizeof(float);
        if (raw_bytes == 0) return false;

        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.element_count);
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, base, raw_bytes };
        return true;
    }

    int addmm_f32_impl(
        const TensorView2DDesc& result_desc,
        const TensorView2DDesc& src_desc,
        const TensorView2DDesc& m1_desc,
        const TensorView2DDesc& m2_desc,
        float beta,
        float alpha)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(m1_desc, "m1") || !validate_desc(m2_desc, "m2"))
        {
            return 0;
        }

        if (beta != 0.0f && !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        const int rows = result_desc.dim0;
        const int cols = result_desc.dim1;
        const int shared = m1_desc.dim1;

        if (m1_desc.dim0 != rows || m2_desc.dim0 != shared || m2_desc.dim1 != cols)
        {
            set_last_error("Size mismatch passed to ggml addmm.");
            return 0;
        }

        if (beta != 0.0f && ((rows % src_desc.dim0) != 0 || (cols % src_desc.dim1) != 0))
        {
            set_last_error("Source tensor shape cannot be broadcast to result shape for ggml addmm.");
            return 0;
        }

        if (!can_map_standard_view(result_desc))
        {
            set_last_error("Result tensor layout is not supported by the ggml addmm Metal path.");
            return 0;
        }

        if (beta != 0.0f && !can_map_standard_view(src_desc))
        {
            set_last_error("Source tensor layout is not supported by the ggml addmm Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(m1_desc) && can_map_m2_direct(m2_desc);

        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        std::vector<BufferHandle> host_ptr_buffers;
        TensorBinding result_binding;
        TensorBinding m1_binding;
        TensorBinding src_binding;
        std::vector<float> packed_m1;
        std::vector<float> packed_m2;

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            result_binding = create_standard_binding(context.value, result_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, m1_desc, m1_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        else
            m1_binding = can_map_standard_view(m1_desc)
                ? create_standard_binding(context.value, m1_desc)
                : create_packed_standard_binding(context.value, m1_desc, packed_m1);

        if (use_zero_copy && beta != 0.0f)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        else if (beta != 0.0f)
            src_binding = create_standard_binding(context.value, src_desc);

        TensorBinding m2_binding;
        bool m2_zero_copy = false;
        if (use_zero_copy && can_map_m2_direct(m2_desc))
        {
            ggml_backend_buffer_t buf = nullptr;
            if (create_binding_from_host_ptr_direct_m2_2d(context.value, g_backend, m2_desc, m2_binding, buf))
            {
                m2_zero_copy = true;
                host_ptr_buffers.emplace_back(buf);
            }
        }
        if (!m2_zero_copy)
            m2_binding = can_map_m2_direct(m2_desc)
                ? create_direct_m2_binding(context.value, m2_desc)
                : create_packed_m2_binding(context.value, m2_desc, packed_m2);

        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            m1_binding.storage == nullptr || m1_binding.tensor == nullptr ||
            m2_binding.storage == nullptr || m2_binding.tensor == nullptr ||
            (beta != 0.0f && (src_binding.storage == nullptr || src_binding.tensor == nullptr)))
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* mm_tensor = ggml_mul_mat(context.value, m2_binding.tensor, m1_binding.tensor);
        if (mm_tensor == nullptr)
        {
            set_last_error("Failed to create ggml matmul node.");
            return 0;
        }

        ggml_tensor* combined_tensor = mm_tensor;
        if (alpha != 1.0f)
        {
            combined_tensor = ggml_scale(context.value, combined_tensor, alpha);
            if (combined_tensor == nullptr)
            {
                set_last_error("Failed to scale ggml matmul output.");
                return 0;
            }
        }

        if (beta != 0.0f)
        {
            ggml_tensor* scaled_src = src_binding.tensor;
            if (beta != 1.0f)
            {
                scaled_src = ggml_scale(context.value, src_binding.tensor, beta);
                if (scaled_src == nullptr)
                {
                    set_last_error("Failed to scale ggml source tensor.");
                    return 0;
                }
            }

            combined_tensor = ggml_add(context.value, combined_tensor, scaled_src);
            if (combined_tensor == nullptr)
            {
                set_last_error("Failed to create ggml add node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, combined_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            if (beta != 0.0f)
                upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (packed_m1.empty())
                upload_binding(m1_binding, m1_desc.data, m1_binding.raw_bytes);
            else
                upload_binding(m1_binding, packed_m1.data(), m1_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }
        if (!m2_zero_copy)
        {
            if (packed_m2.empty())
                upload_binding(m2_binding, m2_desc.data, m2_binding.raw_bytes);
            else
                upload_binding(m2_binding, packed_m2.data(), m2_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int addmm_batch_f32_impl(
        const TensorView3DDesc& result_desc,
        const TensorView3DDesc& src_desc,
        const TensorView3DDesc& m1_desc,
        const TensorView3DDesc& m2_desc,
        float beta,
        float alpha)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(m1_desc, "m1") || !validate_desc(m2_desc, "m2"))
        {
            return 0;
        }

        if (beta != 0.0f && !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        const int batches = result_desc.dim0;
        const int rows = result_desc.dim1;
        const int cols = result_desc.dim2;
        const int shared = m1_desc.dim2;

        if (m1_desc.dim0 != batches || m2_desc.dim0 != batches || m1_desc.dim1 != rows || m2_desc.dim1 != shared || m2_desc.dim2 != cols)
        {
            set_last_error("Size mismatch passed to ggml addmmbatch.");
            return 0;
        }

        if (beta != 0.0f && ((batches % src_desc.dim0) != 0 || (rows % src_desc.dim1) != 0 || (cols % src_desc.dim2) != 0))
        {
            set_last_error("Source tensor shape cannot be broadcast to result shape for ggml addmmbatch.");
            return 0;
        }

        if (!can_map_standard_view(result_desc))
        {
            set_last_error("Result tensor layout is not supported by the ggml addmmbatch Metal path.");
            return 0;
        }

        if (beta != 0.0f && !can_map_standard_view(src_desc))
        {
            set_last_error("Source tensor layout is not supported by the ggml addmmbatch Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(m1_desc) && can_map_m2_direct(m2_desc);

        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        std::vector<BufferHandle> host_ptr_buffers;
        TensorBinding result_binding;
        TensorBinding m1_binding;
        TensorBinding src_binding;
        std::vector<float> packed_m1;
        std::vector<float> packed_m2;
        TensorBinding m2_binding;
        bool m2_zero_copy = false;

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            result_binding = create_standard_binding(context.value, result_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, m1_desc, m1_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        else
            m1_binding = can_map_standard_view(m1_desc)
                ? create_standard_binding(context.value, m1_desc)
                : create_packed_standard_binding(context.value, m1_desc, packed_m1);

        if (use_zero_copy && beta != 0.0f)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        else if (beta != 0.0f)
            src_binding = create_standard_binding(context.value, src_desc);

        if (use_zero_copy && can_map_m2_direct(m2_desc))
        {
            ggml_backend_buffer_t buf = nullptr;
            if (create_binding_from_host_ptr_direct_m2_3d(context.value, g_backend, m2_desc, m2_binding, buf))
            {
                m2_zero_copy = true;
                host_ptr_buffers.emplace_back(buf);
            }
        }
        if (!m2_zero_copy)
            m2_binding = can_map_m2_direct(m2_desc)
                ? create_direct_m2_binding(context.value, m2_desc)
                : create_packed_m2_binding(context.value, m2_desc, packed_m2);

        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            m1_binding.storage == nullptr || m1_binding.tensor == nullptr ||
            m2_binding.storage == nullptr || m2_binding.tensor == nullptr ||
            (beta != 0.0f && (src_binding.storage == nullptr || src_binding.tensor == nullptr)))
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* mm_tensor = ggml_mul_mat(context.value, m2_binding.tensor, m1_binding.tensor);
        if (mm_tensor == nullptr)
        {
            set_last_error("Failed to create ggml batched matmul node.");
            return 0;
        }

        ggml_tensor* combined_tensor = mm_tensor;
        if (alpha != 1.0f)
        {
            combined_tensor = ggml_scale(context.value, combined_tensor, alpha);
            if (combined_tensor == nullptr)
            {
                set_last_error("Failed to scale ggml batched matmul output.");
                return 0;
            }
        }

        if (beta != 0.0f)
        {
            ggml_tensor* scaled_src = src_binding.tensor;
            if (beta != 1.0f)
            {
                scaled_src = ggml_scale(context.value, src_binding.tensor, beta);
                if (scaled_src == nullptr)
                {
                    set_last_error("Failed to scale ggml batched source tensor.");
                    return 0;
                }
            }

            combined_tensor = ggml_add(context.value, combined_tensor, scaled_src);
            if (combined_tensor == nullptr)
            {
                set_last_error("Failed to create ggml batched add node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, combined_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml batched output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            if (beta != 0.0f)
                upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (packed_m1.empty())
                upload_binding(m1_binding, m1_desc.data, m1_binding.raw_bytes);
            else
                upload_binding(m1_binding, packed_m1.data(), m1_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }
        if (!m2_zero_copy)
        {
            if (packed_m2.empty())
                upload_binding(m2_binding, m2_desc.data, m2_binding.raw_bytes);
            else
                upload_binding(m2_binding, packed_m2.data(), m2_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    bool same_shape(const TensorView4DDesc& lhs, const TensorView4DDesc& rhs)
    {
        return lhs.ne0 == rhs.ne0 &&
            lhs.ne1 == rhs.ne1 &&
            lhs.ne2 == rhs.ne2 &&
            lhs.ne3 == rhs.ne3;
    }

    bool same_shape_with_last_dim_reduced(const TensorView4DDesc& result, const TensorView4DDesc& src)
    {
        return result.ne0 == 1 &&
            result.ne1 == src.ne1 &&
            result.ne2 == src.ne2 &&
            result.ne3 == src.ne3;
    }

    bool can_repeat(const TensorView4DDesc& repeated, const TensorView4DDesc& target)
    {
        return (target.ne0 % repeated.ne0) == 0 &&
            (target.ne1 % repeated.ne1) == 0 &&
            (target.ne2 % repeated.ne2) == 0 &&
            (target.ne3 % repeated.ne3) == 0;
    }

    TensorBinding create_scalar_binding(ggml_context* ctx)
    {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        return { tensor, tensor, sizeof(float) };
    }

    TensorBinding create_matrix_binding(ggml_context* ctx, std::int64_t cols, std::int64_t rows)
    {
        ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        return { tensor, tensor, static_cast<std::size_t>(cols * rows * static_cast<std::int64_t>(sizeof(float))) };
    }

    bool build_cross_entropy_label_buffer(
        std::vector<float>& labels,
        const ContiguousTensorDesc& target_indices_desc,
        std::int64_t rows,
        std::int64_t cols,
        float label_smooth)
    {
        if (target_indices_desc.element_count != rows)
        {
            set_last_error("Target index count must match the number of probability rows for ggml crossentropyloss.");
            return false;
        }

        const float base_value = label_smooth > 0.0f
            ? (label_smooth / static_cast<float>(cols))
            : 0.0f;
        const float target_value = 1.0f - label_smooth + (label_smooth / static_cast<float>(cols));

        labels.assign(static_cast<std::size_t>(rows * cols), base_value);

        const float* target_indices = static_cast<const float*>(target_indices_desc.data);
        for (std::int64_t row = 0; row < rows; ++row)
        {
            const std::int64_t target_index = static_cast<std::int64_t>(target_indices[row]);
            if (target_index < 0 || target_index >= cols)
            {
                set_last_error("Target index out of range for ggml crossentropyloss.");
                return false;
            }

            labels[static_cast<std::size_t>(row * cols + target_index)] = target_value;
        }

        return true;
    }

    ggml_tensor* make_unary_tensor(ggml_context* ctx, UnaryOpCode op, ggml_tensor* src)
    {
        switch (op)
        {
        case UnaryOpCode::Neg:
            return ggml_neg(ctx, src);
        case UnaryOpCode::Exp:
            return ggml_exp(ctx, src);
        case UnaryOpCode::Log:
            return ggml_log(ctx, src);
        case UnaryOpCode::Sqrt:
            return ggml_sqrt(ctx, src);
        case UnaryOpCode::Relu:
            return ggml_relu(ctx, src);
        case UnaryOpCode::Sigmoid:
            return ggml_sigmoid(ctx, src);
        case UnaryOpCode::Tanh:
            return ggml_tanh(ctx, src);
        case UnaryOpCode::SiLU:
            return ggml_silu(ctx, src);
        case UnaryOpCode::Step:
            return ggml_step(ctx, src);
        case UnaryOpCode::Abs:
            return ggml_abs(ctx, src);
        case UnaryOpCode::Sign:
            return ggml_sgn(ctx, src);
        default:
            set_last_error("Unsupported unary ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_binary_tensor(ggml_context* ctx, BinaryTensorOpCode op, ggml_tensor* lhs, ggml_tensor* rhs)
    {
        switch (op)
        {
        case BinaryTensorOpCode::Add:
            return ggml_add(ctx, lhs, rhs);
        case BinaryTensorOpCode::Sub:
            return ggml_sub(ctx, lhs, rhs);
        case BinaryTensorOpCode::Mul:
            return ggml_mul(ctx, lhs, rhs);
        case BinaryTensorOpCode::Div:
            return ggml_div(ctx, lhs, rhs);
        default:
            set_last_error("Unsupported binary ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_norm_tensor(ggml_context* ctx, NormOpCode op, ggml_tensor* src, float eps)
    {
        switch (op)
        {
        case NormOpCode::LayerNorm:
            return ggml_norm(ctx, src, eps);
        case NormOpCode::RmsNorm:
            return ggml_rms_norm(ctx, src, eps);
        default:
            set_last_error("Unsupported norm ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_reduction_tensor(ggml_context* ctx, ReductionOpCode op, ggml_tensor* src)
    {
        switch (op)
        {
        case ReductionOpCode::Sum:
            return ggml_sum_rows(ctx, src);
        case ReductionOpCode::Mean:
            return ggml_mean(ctx, src);
        default:
            set_last_error("Unsupported reduction ggml op code.");
            return nullptr;
        }
    }

    bool is_vector_like(const TensorView4DDesc& desc, std::int64_t width)
    {
        return desc.ne0 == width && desc.ne1 == 1 && desc.ne2 == 1 && desc.ne3 == 1;
    }

    std::int64_t flat_row_count(const TensorView4DDesc& desc)
    {
        return static_cast<std::int64_t>(desc.ne1) * desc.ne2 * desc.ne3;
    }

    ggml_tensor* flatten_to_rows(ggml_context* ctx, ggml_tensor* tensor, std::int64_t cols, std::int64_t rows)
    {
        return ggml_reshape_2d(ctx, tensor, cols, rows);
    }

    ggml_tensor* sum_rows_to_feature_vector(ggml_context* ctx, ggml_tensor* tensor)
    {
        ggml_tensor* transposed = ggml_transpose(ctx, tensor);
        ggml_tensor* transposed_contiguous = transposed == nullptr ? nullptr : ggml_cont(ctx, transposed);
        ggml_tensor* summed = transposed_contiguous == nullptr ? nullptr : ggml_sum_rows(ctx, transposed_contiguous);
        ggml_tensor* restored = summed == nullptr ? nullptr : ggml_transpose(ctx, summed);
        return restored == nullptr ? nullptr : ggml_cont(ctx, restored);
    }

    int reduce_last_dim_f32_impl(
        ReductionOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape_with_last_dim_reduced(result_desc, src_desc))
        {
            set_last_error("Result tensor shape must match source shape with the last dimension reduced to 1.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml reduction Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous reduction input.");
            return 0;
        }

        ggml_tensor* reduced_tensor = make_reduction_tensor(context.value, op, contiguous_src);
        if (reduced_tensor == nullptr)
        {
            if (g_last_error.empty())
            {
                set_last_error("Failed to create ggml reduction node.");
            }
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, reduced_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml reduction output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int index_reduction_f32_impl(
        IndexReductionOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape_with_last_dim_reduced(result_desc, src_desc))
        {
            set_last_error("Result tensor shape must match source shape with the last dimension reduced to 1.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml index-reduction Metal path.");
            return 0;
        }

        bool src_zero_copy = can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding src_binding;
        if (src_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                src_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!src_zero_copy)
            src_binding = create_standard_binding(context.value, src_desc);
        if (src_binding.storage == nullptr || src_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous index-reduction input.");
            return 0;
        }

        ggml_tensor* reduction_input = contiguous_src;
        if (op == IndexReductionOpCode::Argmin)
        {
            reduction_input = ggml_neg(context.value, contiguous_src);
            if (reduction_input == nullptr)
            {
                set_last_error("Failed to create ggml argmin preprocessing node.");
                return 0;
            }
        }
        else if (op != IndexReductionOpCode::Argmax)
        {
            set_last_error("Unsupported index-reduction ggml op code.");
            return 0;
        }

        const std::int64_t rows = flat_row_count(src_desc);
        ggml_tensor* flat_input = flatten_to_rows(context.value, reduction_input, src_desc.ne0, rows);
        ggml_tensor* arg_tensor = flat_input == nullptr ? nullptr : ggml_argmax(context.value, flat_input);
        if (flat_input == nullptr || arg_tensor == nullptr)
        {
            set_last_error("Failed to create ggml index-reduction node.");
            return 0;
        }

        ggml_set_output(arg_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, arg_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!src_zero_copy)
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        std::vector<std::int32_t> host_indices(static_cast<std::size_t>(rows));
        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(arg_tensor, host_indices.data(), 0, host_indices.size() * sizeof(std::int32_t));

        float* result_data = static_cast<float*>(result_desc.data);
        for (std::size_t i = 0; i < host_indices.size(); ++i)
        {
            result_data[i] = static_cast<float>(host_indices[i]);
        }

        clear_last_error();
        return 1;
    }

    int copy_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Source tensor shape does not match result shape for ggml copy.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml copy Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous copy input.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, contiguous_src, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int unary_f32_impl(
        UnaryOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Source tensor shape does not match result shape for unary ggml op.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the unary ggml Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous unary input.");
            return 0;
        }

        ggml_tensor* value_tensor = make_unary_tensor(context.value, op, contiguous_src);
        if (value_tensor == nullptr)
        {
            if (g_last_error.empty())
            {
                set_last_error("Failed to create ggml unary node.");
            }
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml unary output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int binary_tensor_f32_impl(
        BinaryTensorOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& lhs_desc,
        const TensorView4DDesc& rhs_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(lhs_desc, "lhs") || !validate_desc(rhs_desc, "rhs"))
        {
            return 0;
        }

        if (!same_shape(result_desc, lhs_desc))
        {
            set_last_error("Result tensor shape does not match lhs tensor shape.");
            return 0;
        }

        if (!can_repeat(rhs_desc, lhs_desc))
        {
            set_last_error("rhs tensor shape cannot be broadcast to lhs for ggml binary op.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(lhs_desc) || !can_map_standard_view(rhs_desc))
        {
            set_last_error("Tensor layout is not supported by the binary ggml Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(lhs_desc) && can_map_standard_view(rhs_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding lhs_binding;
        TensorBinding rhs_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, lhs_desc, lhs_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, rhs_desc, rhs_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            lhs_binding = create_standard_binding(context.value, lhs_desc);
            rhs_binding = create_standard_binding(context.value, rhs_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            lhs_binding.storage == nullptr || lhs_binding.tensor == nullptr ||
            rhs_binding.storage == nullptr || rhs_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* value_tensor = make_binary_tensor(context.value, op, lhs_binding.tensor, rhs_binding.tensor);
        if (value_tensor == nullptr)
        {
            if (g_last_error.empty())
            {
                set_last_error("Failed to create ggml binary node.");
            }
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml binary output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(lhs_binding, lhs_desc.data, lhs_binding.raw_bytes);
            upload_binding(rhs_binding, rhs_desc.data, rhs_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int binary_scalar_f32_impl(
        BinaryScalarOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc,
        float scalar)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Result tensor shape does not match source tensor shape for scalar ggml op.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the scalar ggml Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        TensorBinding scalar_binding = create_scalar_binding(context.value);
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            scalar_binding.storage == nullptr || scalar_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous scalar-op input.");
            return 0;
        }

        ggml_tensor* value_tensor = nullptr;
        if (op == BinaryScalarOpCode::Mul)
        {
            value_tensor = ggml_scale(context.value, contiguous_src, scalar);
        }
        else
        {
            ggml_tensor* repeated_scalar = ggml_repeat(context.value, scalar_binding.tensor, contiguous_src);
            if (repeated_scalar == nullptr)
            {
                set_last_error("Failed to create repeated scalar tensor.");
                return 0;
            }

            switch (op)
            {
            case BinaryScalarOpCode::Add:
                value_tensor = ggml_add(context.value, contiguous_src, repeated_scalar);
                break;
            case BinaryScalarOpCode::Sub:
                value_tensor = ggml_sub(context.value, contiguous_src, repeated_scalar);
                break;
            case BinaryScalarOpCode::ReverseSub:
                value_tensor = ggml_sub(context.value, repeated_scalar, contiguous_src);
                break;
            case BinaryScalarOpCode::Div:
                value_tensor = ggml_div(context.value, contiguous_src, repeated_scalar);
                break;
            case BinaryScalarOpCode::ReverseDiv:
                value_tensor = ggml_div(context.value, repeated_scalar, contiguous_src);
                break;
            default:
                set_last_error("Unsupported scalar ggml op code.");
                return 0;
            }
        }

        if (value_tensor == nullptr)
        {
            if (g_last_error.empty())
            {
                set_last_error("Failed to create ggml scalar op node.");
            }
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml scalar-op output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }
        if (op != BinaryScalarOpCode::Mul)
            ggml_backend_tensor_set(scalar_binding.storage, &scalar, 0, sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int activation_grad_f32_impl(
        ActivationGradOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc,
        const TensorView4DDesc& grad_desc,
        const TensorView4DDesc& accumulation_desc,
        bool has_accumulation)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src") || !validate_desc(grad_desc, "grad"))
        {
            return 0;
        }

        if (has_accumulation && !validate_desc(accumulation_desc, "accumulation"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc) || !same_shape(src_desc, grad_desc) ||
            (has_accumulation && !same_shape(src_desc, accumulation_desc)))
        {
            set_last_error("Tensor shape mismatch passed to ggml activation grad.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc) || !can_map_standard_view(grad_desc) ||
            (has_accumulation && !can_map_standard_view(accumulation_desc)))
        {
            set_last_error("Tensor layout is not supported by the ggml activation-grad Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc) && can_map_standard_view(grad_desc) &&
            (!has_accumulation || can_map_standard_view(accumulation_desc));
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        TensorBinding grad_binding;
        TensorBinding accumulation_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, grad_desc, grad_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy && has_accumulation)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, accumulation_desc, accumulation_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
            grad_binding = create_standard_binding(context.value, grad_desc);
            if (has_accumulation)
                accumulation_binding = create_standard_binding(context.value, accumulation_desc);
        }
        TensorBinding one_binding = create_scalar_binding(context.value);
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            grad_binding.storage == nullptr || grad_binding.tensor == nullptr ||
            one_binding.storage == nullptr || one_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        if (has_accumulation && (accumulation_binding.storage == nullptr || accumulation_binding.tensor == nullptr))
        {
            set_last_error("Failed to allocate ggml accumulation tensor.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        ggml_tensor* contiguous_grad = ggml_cont(context.value, grad_binding.tensor);
        if (contiguous_src == nullptr || contiguous_grad == nullptr)
        {
            set_last_error("Failed to create ggml contiguous activation-grad inputs.");
            return 0;
        }

        ggml_tensor* value_tensor = nullptr;
        switch (op)
        {
        case ActivationGradOpCode::Relu:
        {
            ggml_tensor* step_tensor = ggml_step(context.value, contiguous_src);
            if (step_tensor != nullptr)
            {
                value_tensor = ggml_mul(context.value, step_tensor, contiguous_grad);
            }
        } break;
        case ActivationGradOpCode::Sigmoid:
        {
            ggml_tensor* one_tensor = ggml_repeat(context.value, one_binding.tensor, contiguous_src);
            ggml_tensor* one_minus = one_tensor == nullptr ? nullptr : ggml_sub(context.value, one_tensor, contiguous_src);
            ggml_tensor* deriv_tensor = one_minus == nullptr ? nullptr : ggml_mul(context.value, contiguous_src, one_minus);
            value_tensor = deriv_tensor == nullptr ? nullptr : ggml_mul(context.value, deriv_tensor, contiguous_grad);
        } break;
        case ActivationGradOpCode::Tanh:
        {
            ggml_tensor* one_tensor = ggml_repeat(context.value, one_binding.tensor, contiguous_src);
            ggml_tensor* sq_tensor = ggml_mul(context.value, contiguous_src, contiguous_src);
            ggml_tensor* one_minus = (one_tensor == nullptr || sq_tensor == nullptr) ? nullptr : ggml_sub(context.value, one_tensor, sq_tensor);
            value_tensor = one_minus == nullptr ? nullptr : ggml_mul(context.value, one_minus, contiguous_grad);
        } break;
        case ActivationGradOpCode::SiLU:
        {
            value_tensor = ggml_silu_back(context.value, contiguous_grad, contiguous_src);
            if (!backend_supports_op(value_tensor))
            {
                ggml_tensor* one_tensor = ggml_repeat(context.value, one_binding.tensor, contiguous_src);
                ggml_tensor* sig_tensor = ggml_sigmoid(context.value, contiguous_src);
                ggml_tensor* one_minus_sig = (one_tensor == nullptr || sig_tensor == nullptr) ? nullptr : ggml_sub(context.value, one_tensor, sig_tensor);
                ggml_tensor* weighted_tensor = one_minus_sig == nullptr ? nullptr : ggml_mul(context.value, contiguous_src, one_minus_sig);
                ggml_tensor* inner_tensor = (one_tensor == nullptr || weighted_tensor == nullptr) ? nullptr : ggml_add(context.value, one_tensor, weighted_tensor);
                ggml_tensor* deriv_tensor = (sig_tensor == nullptr || inner_tensor == nullptr) ? nullptr : ggml_mul(context.value, sig_tensor, inner_tensor);
                value_tensor = deriv_tensor == nullptr ? nullptr : ggml_mul(context.value, deriv_tensor, contiguous_grad);
            }
        } break;
        default:
            set_last_error("Unsupported activation-grad ggml op code.");
            return 0;
        }

        if (value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml activation-grad node.");
            return 0;
        }

        if (has_accumulation)
        {
            ggml_tensor* contiguous_accumulation = ggml_cont(context.value, accumulation_binding.tensor);
            if (contiguous_accumulation == nullptr)
            {
                set_last_error("Failed to create ggml contiguous accumulation input.");
                return 0;
            }

            value_tensor = ggml_add(context.value, contiguous_accumulation, value_tensor);
            if (value_tensor == nullptr)
            {
                set_last_error("Failed to create ggml activation-grad accumulation node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml activation-grad output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            upload_binding(grad_binding, grad_desc.data, grad_binding.raw_bytes);
            if (has_accumulation)
                upload_binding(accumulation_binding, accumulation_desc.data, accumulation_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }
        const float one_value = 1.0f;
        ggml_backend_tensor_set(one_binding.storage, &one_value, 0, sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int norm_f32_impl(
        NormOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc,
        const TensorView4DDesc& gamma_desc,
        const TensorView4DDesc& beta_desc,
        bool has_beta,
        float eps)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src") || !validate_desc(gamma_desc, "gamma"))
        {
            return 0;
        }

        if (has_beta && !validate_desc(beta_desc, "beta"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Result tensor shape does not match source tensor shape for ggml norm op.");
            return 0;
        }

        if (!can_repeat(gamma_desc, src_desc) || (has_beta && !can_repeat(beta_desc, src_desc)))
        {
            set_last_error("gamma/beta tensor shape cannot be broadcast to source tensor for ggml norm op.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc) || !can_map_standard_view(gamma_desc) ||
            (has_beta && !can_map_standard_view(beta_desc)))
        {
            set_last_error("Tensor layout is not supported by the ggml norm Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc) &&
            can_map_standard_view(gamma_desc) && (!has_beta || can_map_standard_view(beta_desc));
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 3 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        TensorBinding gamma_binding;
        TensorBinding beta_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, gamma_desc, gamma_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy && has_beta)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, beta_desc, beta_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
            gamma_binding = create_standard_binding(context.value, gamma_desc);
            if (has_beta)
                beta_binding = create_standard_binding(context.value, beta_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            gamma_binding.storage == nullptr || gamma_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        if (has_beta && (beta_binding.storage == nullptr || beta_binding.tensor == nullptr))
        {
            set_last_error("Failed to allocate ggml beta tensor.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        ggml_tensor* contiguous_gamma = ggml_cont(context.value, gamma_binding.tensor);
        if (contiguous_src == nullptr || contiguous_gamma == nullptr)
        {
            set_last_error("Failed to create ggml contiguous norm inputs.");
            return 0;
        }

        ggml_tensor* value_tensor = make_norm_tensor(context.value, op, contiguous_src, eps);
        if (value_tensor == nullptr)
        {
            if (g_last_error.empty())
            {
                set_last_error("Failed to create ggml norm node.");
            }
            return 0;
        }

        value_tensor = ggml_mul(context.value, value_tensor, contiguous_gamma);
        if (value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml norm scale node.");
            return 0;
        }

        if (has_beta)
        {
            ggml_tensor* contiguous_beta = ggml_cont(context.value, beta_binding.tensor);
            if (contiguous_beta == nullptr)
            {
                set_last_error("Failed to create ggml contiguous beta tensor.");
                return 0;
            }

            value_tensor = ggml_add(context.value, value_tensor, contiguous_beta);
            if (value_tensor == nullptr)
            {
                set_last_error("Failed to create ggml norm bias node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml norm output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            upload_binding(gamma_binding, gamma_desc.data, gamma_binding.raw_bytes);
            if (has_beta)
                upload_binding(beta_binding, beta_desc.data, beta_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int norm_grad_f32_impl(
        NormOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& grad_gamma_desc,
        const TensorView4DDesc& grad_beta_desc,
        const TensorView4DDesc& adj_desc,
        const TensorView4DDesc& x_desc,
        const TensorView4DDesc& gamma_desc,
        bool has_grad_beta,
        float eps)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result")
            || !validate_desc(grad_gamma_desc, "gradGamma")
            || !validate_desc(adj_desc, "adj")
            || !validate_desc(x_desc, "x")
            || !validate_desc(gamma_desc, "gamma"))
        {
            return 0;
        }

        if (has_grad_beta && !validate_desc(grad_beta_desc, "gradBeta"))
        {
            return 0;
        }

        if (!same_shape(result_desc, adj_desc) || !same_shape(adj_desc, x_desc))
        {
            set_last_error("Tensor shape mismatch passed to ggml norm grad.");
            return 0;
        }

        if (!is_vector_like(gamma_desc, x_desc.ne0) || !is_vector_like(grad_gamma_desc, x_desc.ne0) || (has_grad_beta && !is_vector_like(grad_beta_desc, x_desc.ne0)))
        {
            set_last_error("gamma/gradGamma/gradBeta must match the last source dimension for ggml norm grad.");
            return 0;
        }

        if (!can_map_standard_view(result_desc)
            || !can_map_standard_view(grad_gamma_desc)
            || !can_map_standard_view(adj_desc)
            || !can_map_standard_view(x_desc)
            || !can_map_standard_view(gamma_desc)
            || (has_grad_beta && !can_map_standard_view(grad_beta_desc)))
        {
            set_last_error("Tensor layout is not supported by the ggml norm-grad Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(grad_gamma_desc) &&
            can_map_standard_view(adj_desc) && can_map_standard_view(x_desc) && can_map_standard_view(gamma_desc) &&
            (!has_grad_beta || can_map_standard_view(grad_beta_desc));
        std::vector<BufferHandle> host_ptr_buffers;
        constexpr size_t graph_capacity = 512;
        const std::size_t ctx_size = 16 * 1024 * 1024 + ggml_graph_overhead_custom(graph_capacity, true);

        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding grad_gamma_binding;
        TensorBinding adj_binding;
        TensorBinding x_binding;
        TensorBinding gamma_binding;
        TensorBinding grad_beta_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, grad_gamma_desc, grad_gamma_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, adj_desc, adj_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, x_desc, x_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, gamma_desc, gamma_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy && has_grad_beta)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, grad_beta_desc, grad_beta_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            grad_gamma_binding = create_standard_binding(context.value, grad_gamma_desc);
            adj_binding = create_standard_binding(context.value, adj_desc);
            x_binding = create_standard_binding(context.value, x_desc);
            gamma_binding = create_standard_binding(context.value, gamma_desc);
            if (has_grad_beta)
                grad_beta_binding = create_standard_binding(context.value, grad_beta_desc);
        }
        TensorBinding eps_binding = create_scalar_binding(context.value);

        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            grad_gamma_binding.storage == nullptr || grad_gamma_binding.tensor == nullptr ||
            adj_binding.storage == nullptr || adj_binding.tensor == nullptr ||
            x_binding.storage == nullptr || x_binding.tensor == nullptr ||
            gamma_binding.storage == nullptr || gamma_binding.tensor == nullptr ||
            eps_binding.storage == nullptr || eps_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        if (has_grad_beta && (grad_beta_binding.storage == nullptr || grad_beta_binding.tensor == nullptr))
        {
            set_last_error("Failed to allocate ggml gradBeta tensor.");
            return 0;
        }

        ggml_tensor* contiguous_result = ggml_cont(context.value, result_binding.tensor);
        ggml_tensor* contiguous_grad_gamma = ggml_cont(context.value, grad_gamma_binding.tensor);
        ggml_tensor* contiguous_adj = ggml_cont(context.value, adj_binding.tensor);
        ggml_tensor* contiguous_x = ggml_cont(context.value, x_binding.tensor);
        ggml_tensor* contiguous_gamma = ggml_cont(context.value, gamma_binding.tensor);
        ggml_tensor* contiguous_grad_beta = nullptr;
        if (has_grad_beta)
        {
            contiguous_grad_beta = ggml_cont(context.value, grad_beta_binding.tensor);
        }

        if (contiguous_result == nullptr || contiguous_grad_gamma == nullptr || contiguous_adj == nullptr || contiguous_x == nullptr || contiguous_gamma == nullptr ||
            (has_grad_beta && contiguous_grad_beta == nullptr))
        {
            set_last_error("Failed to create ggml contiguous norm-grad inputs.");
            return 0;
        }

        if (op == NormOpCode::LayerNorm)
        {
            ggml_set_param(x_binding.storage);

            ggml_tensor* norm_value = ggml_norm(context.value, contiguous_x, eps);
            ggml_tensor* scaled_value = norm_value == nullptr ? nullptr : ggml_mul(context.value, norm_value, contiguous_gamma);
            ggml_tensor* weighted_value = scaled_value == nullptr ? nullptr : ggml_mul(context.value, scaled_value, contiguous_adj);
            ggml_tensor* loss_tensor = weighted_value == nullptr ? nullptr : ggml_sum(context.value, weighted_value);
            if (loss_tensor == nullptr)
            {
                set_last_error("Failed to create ggml layernorm backward loss graph.");
                return 0;
            }
            ggml_set_loss(loss_tensor);

            ggml_cgraph* graph = ggml_new_graph_custom(context.value, graph_capacity, true);
            if (graph == nullptr)
            {
                set_last_error("Failed to create ggml backward graph.");
                return 0;
            }

            ggml_build_forward_expand(graph, loss_tensor);
            ggml_build_backward_expand(context.value, graph, nullptr);

            ggml_tensor* dx_delta = ggml_graph_get_grad(graph, contiguous_x);
            if (dx_delta == nullptr)
            {
                set_last_error("Failed to obtain ggml layernorm input gradient.");
                return 0;
            }

            const std::int64_t rows = flat_row_count(x_desc);
            ggml_tensor* flat_adj = flatten_to_rows(context.value, contiguous_adj, x_desc.ne0, rows);
            ggml_tensor* flat_norm = norm_value == nullptr ? nullptr : flatten_to_rows(context.value, norm_value, x_desc.ne0, rows);
            ggml_tensor* flat_grad_gamma = flatten_to_rows(context.value, contiguous_grad_gamma, x_desc.ne0, 1);
            ggml_tensor* flat_grad_beta = has_grad_beta ? flatten_to_rows(context.value, contiguous_grad_beta, x_desc.ne0, 1) : nullptr;
            if (flat_adj == nullptr || flat_norm == nullptr || flat_grad_gamma == nullptr || (has_grad_beta && flat_grad_beta == nullptr))
            {
                set_last_error("Failed to reshape ggml layernorm gradient tensors.");
                return 0;
            }

            ggml_tensor* adj_norm = ggml_mul(context.value, flat_adj, flat_norm);
            ggml_tensor* grad_gamma_delta = adj_norm == nullptr ? nullptr : sum_rows_to_feature_vector(context.value, adj_norm);
            ggml_tensor* grad_beta_delta = has_grad_beta ? sum_rows_to_feature_vector(context.value, flat_adj) : nullptr;
            if (grad_gamma_delta == nullptr || (has_grad_beta && grad_beta_delta == nullptr))
            {
                set_last_error("Failed to create ggml layernorm parameter gradients.");
                return 0;
            }

            ggml_tensor* dx_value = ggml_add(context.value, contiguous_result, dx_delta);
            ggml_tensor* grad_gamma_value = ggml_add(context.value, flat_grad_gamma, grad_gamma_delta);
            ggml_tensor* grad_gamma_view = grad_gamma_value == nullptr ? nullptr : ggml_reshape_4d(context.value, grad_gamma_value, grad_gamma_desc.ne0, grad_gamma_desc.ne1, grad_gamma_desc.ne2, grad_gamma_desc.ne3);
            ggml_tensor* grad_beta_value = has_grad_beta ? ggml_add(context.value, flat_grad_beta, grad_beta_delta) : nullptr;
            ggml_tensor* grad_beta_view = has_grad_beta && grad_beta_value != nullptr
                ? ggml_reshape_4d(context.value, grad_beta_value, grad_beta_desc.ne0, grad_beta_desc.ne1, grad_beta_desc.ne2, grad_beta_desc.ne3)
                : nullptr;
            ggml_tensor* dx_output = dx_value == nullptr ? nullptr : ggml_cpy(context.value, dx_value, result_binding.tensor);
            ggml_tensor* grad_gamma_output = grad_gamma_view == nullptr ? nullptr : ggml_cpy(context.value, grad_gamma_view, grad_gamma_binding.tensor);
            ggml_tensor* grad_beta_output = has_grad_beta
                ? (grad_beta_view == nullptr ? nullptr : ggml_cpy(context.value, grad_beta_view, grad_beta_binding.tensor))
                : nullptr;
            if (dx_output == nullptr || grad_gamma_output == nullptr || (has_grad_beta && grad_beta_output == nullptr))
            {
                set_last_error("Failed to create ggml layernorm output copy nodes.");
                return 0;
            }

            ggml_set_output(dx_output);
            ggml_set_output(grad_gamma_output);
            if (has_grad_beta)
            {
                ggml_set_output(grad_beta_output);
            }

            ggml_build_forward_expand(graph, dx_output);
            ggml_build_forward_expand(graph, grad_gamma_output);
            if (has_grad_beta)
            {
                ggml_build_forward_expand(graph, grad_beta_output);
            }

            BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
            if (buffer.value == nullptr)
            {
                set_last_error("Failed to allocate ggml backend buffer.");
                return 0;
            }

            if (!use_zero_copy)
            {
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
                upload_binding(grad_gamma_binding, grad_gamma_desc.data, grad_gamma_binding.raw_bytes);
                upload_binding(adj_binding, adj_desc.data, adj_binding.raw_bytes);
                upload_binding(x_binding, x_desc.data, x_binding.raw_bytes);
                upload_binding(gamma_binding, gamma_desc.data, gamma_binding.raw_bytes);
                if (has_grad_beta)
                    upload_binding(grad_beta_binding, grad_beta_desc.data, grad_beta_binding.raw_bytes);
            }

            ggml_graph_reset(graph);

            ggml_status status = ggml_backend_graph_compute(g_backend, graph);
            if (status != GGML_STATUS_SUCCESS)
            {
                set_last_error("ggml backend graph execution failed.");
                return 0;
            }

            ggml_backend_synchronize(g_backend);
            if (!use_zero_copy)
            {
                ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);
                ggml_backend_tensor_get(grad_gamma_binding.storage, grad_gamma_desc.data, 0, grad_gamma_binding.raw_bytes);
                if (has_grad_beta)
                    ggml_backend_tensor_get(grad_beta_binding.storage, grad_beta_desc.data, 0, grad_beta_binding.raw_bytes);
            }

            clear_last_error();
            return 1;
        }

        const std::int64_t rows = flat_row_count(x_desc);
        const float inv_cols = 1.0f / static_cast<float>(x_desc.ne0);
        const float cols_value = static_cast<float>(x_desc.ne0);

        ggml_tensor* flat_adj = flatten_to_rows(context.value, contiguous_adj, x_desc.ne0, rows);
        ggml_tensor* flat_x = flatten_to_rows(context.value, contiguous_x, x_desc.ne0, rows);
        ggml_tensor* flat_gamma = flatten_to_rows(context.value, contiguous_gamma, x_desc.ne0, 1);
        ggml_tensor* flat_grad_gamma = flatten_to_rows(context.value, contiguous_grad_gamma, x_desc.ne0, 1);
        ggml_tensor* flat_grad_beta = has_grad_beta ? flatten_to_rows(context.value, contiguous_grad_beta, x_desc.ne0, 1) : nullptr;
        if (flat_adj == nullptr || flat_x == nullptr || flat_gamma == nullptr || flat_grad_gamma == nullptr || (has_grad_beta && flat_grad_beta == nullptr))
        {
            set_last_error("Failed to reshape ggml norm-grad tensors.");
            return 0;
        }

        ggml_tensor* dx_delta_flat = nullptr;
        ggml_tensor* grad_gamma_delta = nullptr;
        ggml_tensor* grad_beta_delta = nullptr;

        switch (op)
        {
        case NormOpCode::RmsNorm:
        {
            ggml_tensor* native_adj = ggml_mul(context.value, contiguous_adj, contiguous_gamma);
            ggml_tensor* native_dx = native_adj == nullptr ? nullptr : ggml_rms_norm_back(context.value, native_adj, contiguous_x, eps);
            if (backend_supports_op(native_dx))
            {
                dx_delta_flat = flatten_to_rows(context.value, native_dx, x_desc.ne0, rows);
            }

            ggml_tensor* sq = ggml_mul(context.value, flat_x, flat_x);
            ggml_tensor* sq_sum = sq == nullptr ? nullptr : ggml_sum_rows(context.value, sq);
            ggml_tensor* mean_sq = sq_sum == nullptr ? nullptr : ggml_scale(context.value, sq_sum, inv_cols);
            ggml_tensor* eps_full = mean_sq == nullptr ? nullptr : ggml_repeat(context.value, eps_binding.tensor, mean_sq);
            ggml_tensor* rms_sq = (mean_sq == nullptr || eps_full == nullptr) ? nullptr : ggml_add(context.value, mean_sq, eps_full);
            ggml_tensor* rms = rms_sq == nullptr ? nullptr : ggml_sqrt(context.value, rms_sq);
            ggml_tensor* rms_full = rms == nullptr ? nullptr : ggml_repeat(context.value, rms, flat_x);
            ggml_tensor* rms_norm = rms_full == nullptr ? nullptr : ggml_div(context.value, flat_x, rms_full);
            ggml_tensor* adj_rms_norm = rms_norm == nullptr ? nullptr : ggml_mul(context.value, flat_adj, rms_norm);
            ggml_tensor* sum_adj_rms_norm = adj_rms_norm == nullptr ? nullptr : ggml_sum_rows(context.value, adj_rms_norm);
            ggml_tensor* sum_adj_rms_norm_full = sum_adj_rms_norm == nullptr ? nullptr : ggml_repeat(context.value, sum_adj_rms_norm, flat_x);
            ggml_tensor* weighted = (rms_norm == nullptr || sum_adj_rms_norm_full == nullptr) ? nullptr : ggml_mul(context.value, rms_norm, sum_adj_rms_norm_full);
            ggml_tensor* scaled_adj = ggml_scale(context.value, flat_adj, cols_value);
            ggml_tensor* dx_numerator = (scaled_adj == nullptr || weighted == nullptr) ? nullptr : ggml_sub(context.value, scaled_adj, weighted);
            ggml_tensor* dx_denominator = rms_full == nullptr ? nullptr : ggml_scale(context.value, rms_full, cols_value);
            ggml_tensor* dx_core = (dx_numerator == nullptr || dx_denominator == nullptr) ? nullptr : ggml_div(context.value, dx_numerator, dx_denominator);
            ggml_tensor* unclamped = (dx_core == nullptr) ? nullptr : ggml_mul(context.value, dx_core, flat_gamma);

            if (dx_delta_flat == nullptr)
            {
                dx_delta_flat = unclamped == nullptr ? nullptr : ggml_clamp(context.value, unclamped, -1000.0f, 1000.0f);
            }
            grad_gamma_delta = adj_rms_norm == nullptr ? nullptr : sum_rows_to_feature_vector(context.value, adj_rms_norm);
            if (has_grad_beta)
            {
                grad_beta_delta = sum_rows_to_feature_vector(context.value, flat_adj);
            }
        } break;
        default:
            set_last_error("Unsupported norm-grad ggml op code.");
            return 0;
        }

        if (dx_delta_flat == nullptr || grad_gamma_delta == nullptr || (has_grad_beta && grad_beta_delta == nullptr))
        {
            set_last_error("Failed to create ggml norm-grad intermediate tensors.");
            return 0;
        }

        ggml_tensor* dx_delta = ggml_reshape_4d(context.value, dx_delta_flat, result_desc.ne0, result_desc.ne1, result_desc.ne2, result_desc.ne3);
        ggml_tensor* dx_value = dx_delta == nullptr ? nullptr : ggml_add(context.value, contiguous_result, dx_delta);
        ggml_tensor* grad_gamma_value = ggml_add(context.value, flat_grad_gamma, grad_gamma_delta);
        ggml_tensor* grad_gamma_view = grad_gamma_value == nullptr ? nullptr : ggml_reshape_4d(context.value, grad_gamma_value, grad_gamma_desc.ne0, grad_gamma_desc.ne1, grad_gamma_desc.ne2, grad_gamma_desc.ne3);
        ggml_tensor* grad_beta_value = nullptr;
        ggml_tensor* grad_beta_view = nullptr;
        if (has_grad_beta)
        {
            grad_beta_value = ggml_add(context.value, flat_grad_beta, grad_beta_delta);
            grad_beta_view = grad_beta_value == nullptr ? nullptr : ggml_reshape_4d(context.value, grad_beta_value, grad_beta_desc.ne0, grad_beta_desc.ne1, grad_beta_desc.ne2, grad_beta_desc.ne3);
        }

        if (dx_value == nullptr || grad_gamma_view == nullptr || (has_grad_beta && grad_beta_view == nullptr))
        {
            set_last_error("Failed to create ggml norm-grad accumulation tensors.");
            return 0;
        }

        ggml_tensor* dx_output = ggml_cpy(context.value, dx_value, result_binding.tensor);
        ggml_tensor* grad_gamma_output = ggml_cpy(context.value, grad_gamma_view, grad_gamma_binding.tensor);
        ggml_tensor* grad_beta_output = has_grad_beta ? ggml_cpy(context.value, grad_beta_view, grad_beta_binding.tensor) : nullptr;
        if (dx_output == nullptr || grad_gamma_output == nullptr || (has_grad_beta && grad_beta_output == nullptr))
        {
            set_last_error("Failed to create ggml norm-grad output copy nodes.");
            return 0;
        }

        ggml_set_output(dx_output);
        ggml_set_output(grad_gamma_output);
        if (has_grad_beta)
        {
            ggml_set_output(grad_beta_output);
        }

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, dx_output);
        ggml_build_forward_expand(graph, grad_gamma_output);
        if (has_grad_beta)
        {
            ggml_build_forward_expand(graph, grad_beta_output);
        }

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
            upload_binding(grad_gamma_binding, grad_gamma_desc.data, grad_gamma_binding.raw_bytes);
            upload_binding(adj_binding, adj_desc.data, adj_binding.raw_bytes);
            upload_binding(x_binding, x_desc.data, x_binding.raw_bytes);
            upload_binding(gamma_binding, gamma_desc.data, gamma_binding.raw_bytes);
            if (has_grad_beta)
                upload_binding(grad_beta_binding, grad_beta_desc.data, grad_beta_binding.raw_bytes);
        }
        ggml_backend_tensor_set(eps_binding.storage, &eps, 0, sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
        {
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);
            ggml_backend_tensor_get(grad_gamma_binding.storage, grad_gamma_desc.data, 0, grad_gamma_binding.raw_bytes);
            if (has_grad_beta)
                ggml_backend_tensor_get(grad_beta_binding.storage, grad_beta_desc.data, 0, grad_beta_binding.raw_bytes);
        }

        clear_last_error();
        return 1;
    }

    int index_select_f32_impl(
        const TensorView2DDesc& result_desc,
        const TensorView2DDesc& src_desc,
        const ContiguousTensorDesc& indices_desc,
        bool add_to_result)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src") || !validate_desc(indices_desc, "indices"))
        {
            return 0;
        }

        if (result_desc.dim1 != src_desc.dim1 || indices_desc.element_count != result_desc.dim0)
        {
            set_last_error("Tensor shape mismatch passed to ggml indexselect.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml indexselect Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        ggml_tensor* index_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I32, indices_desc.element_count);
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            index_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous indexselect input.");
            return 0;
        }

        ggml_tensor* value_tensor = ggml_get_rows(context.value, contiguous_src, index_tensor);
        if (value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml get_rows node.");
            return 0;
        }

        if (add_to_result)
        {
            ggml_tensor* contiguous_result = ggml_cont(context.value, result_binding.tensor);
            if (contiguous_result == nullptr)
            {
                set_last_error("Failed to create ggml contiguous indexselect accumulation input.");
                return 0;
            }

            value_tensor = ggml_add(context.value, value_tensor, contiguous_result);
            if (value_tensor == nullptr)
            {
                set_last_error("Failed to create ggml indexselect accumulation node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml indexselect output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (add_to_result || result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        std::vector<std::int32_t> indices(static_cast<std::size_t>(indices_desc.element_count));
        const float* raw_indices = static_cast<const float*>(indices_desc.data);
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            indices[i] = static_cast<std::int32_t>(raw_indices[i]);
        }
        ggml_backend_tensor_set(index_tensor, indices.data(), 0, indices.size() * sizeof(std::int32_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int index_select_grad_f32_impl(
        const TensorView2DDesc& grad_desc,
        const TensorView2DDesc& adj_desc,
        const ContiguousTensorDesc& indices_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(grad_desc, "grad") || !validate_desc(adj_desc, "adj") || !validate_desc(indices_desc, "indices"))
        {
            return 0;
        }

        if (adj_desc.dim0 != indices_desc.element_count || grad_desc.dim1 != adj_desc.dim1)
        {
            set_last_error("Tensor shape mismatch passed to ggml indexselectgrad.");
            return 0;
        }

        if (!can_map_standard_view(grad_desc) || !can_map_standard_view(adj_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml indexselectgrad Metal path.");
            return 0;
        }

        const float* raw_indices = static_cast<const float*>(indices_desc.data);
        std::vector<std::int32_t> indices(static_cast<std::size_t>(indices_desc.element_count));
        std::size_t active_row_count = 0;
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            indices[i] = static_cast<std::int32_t>(raw_indices[i]);
            if (indices[i] >= 0)
            {
                ++active_row_count;
            }
        }

        bool use_zero_copy = can_map_standard_view(grad_desc) && can_map_standard_view(adj_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t min_graph_capacity = static_cast<std::size_t>(GGML_DEFAULT_GRAPH_SIZE) * 8;
        const std::size_t estimated_graph_capacity = active_row_count * 6 + 64;
        const std::size_t graph_capacity = estimated_graph_capacity > min_graph_capacity ? estimated_graph_capacity : min_graph_capacity;

        const std::size_t ctx_size = 16 * 1024 * 1024 + ggml_graph_overhead_custom(graph_capacity, false);

        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding grad_binding;
        TensorBinding adj_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, grad_desc, grad_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, adj_desc, adj_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            grad_binding = create_standard_binding(context.value, grad_desc);
            adj_binding = create_standard_binding(context.value, adj_desc);
        }
        if (grad_binding.storage == nullptr || grad_binding.tensor == nullptr ||
            adj_binding.storage == nullptr || adj_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        ggml_tensor* working_grad = ggml_cont(context.value, grad_binding.tensor);
        ggml_tensor* contiguous_adj = ggml_cont(context.value, adj_binding.tensor);
        if (working_grad == nullptr || contiguous_adj == nullptr)
        {
            set_last_error("Failed to create ggml contiguous indexselectgrad inputs.");
            return 0;
        }

        struct PendingIndexUpload
        {
            ggml_tensor* tensor;
            std::int32_t value;
        };

        std::vector<PendingIndexUpload> pending_index_uploads;
        pending_index_uploads.reserve(indices.size());

        const std::size_t row_bytes = static_cast<std::size_t>(adj_desc.dim1) * sizeof(float);
        for (std::size_t row = 0; row < indices.size(); ++row)
        {
            if (indices[row] < 0)
            {
                continue;
            }

            ggml_tensor* index_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I32, 1);
            ggml_tensor* current_row = index_tensor == nullptr ? nullptr : ggml_get_rows(context.value, working_grad, index_tensor);
            ggml_tensor* adj_row = current_row == nullptr ? nullptr : ggml_view_2d(
                context.value,
                contiguous_adj,
                adj_desc.dim1,
                1,
                row_bytes,
                row * row_bytes);
            ggml_tensor* updated_row = (current_row == nullptr || adj_row == nullptr) ? nullptr : ggml_add(context.value, current_row, adj_row);
            ggml_tensor* updated_grad = (updated_row == nullptr) ? nullptr : ggml_set_rows(context.value, working_grad, updated_row, index_tensor);

            if (index_tensor == nullptr || current_row == nullptr || adj_row == nullptr || updated_row == nullptr || updated_grad == nullptr)
            {
                set_last_error("Failed to create ggml indexselectgrad scatter-add node.");
                return 0;
            }

            pending_index_uploads.push_back({ index_tensor, indices[row] });
            working_grad = updated_grad;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, working_grad, grad_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml indexselectgrad output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph_custom(context.value, graph_capacity, false);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(grad_binding, grad_desc.data, grad_binding.raw_bytes);
            upload_binding(adj_binding, adj_desc.data, adj_binding.raw_bytes);
        }
        for (const PendingIndexUpload& upload : pending_index_uploads)
        {
            ggml_backend_tensor_set(upload.tensor, &upload.value, 0, sizeof(upload.value));
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(grad_binding.storage, grad_desc.data, 0, grad_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int rope_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc,
        int seq_len,
        int row_offset,
        bool add_to_result,
        bool invert_positions)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (seq_len <= 0)
        {
            set_last_error("seqLen must be positive for ggml rope.");
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Source tensor shape does not match result shape for ggml rope.");
            return 0;
        }

        if ((src_desc.ne0 % 2) != 0)
        {
            set_last_error("ggml rope requires an even embedding dimension.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml rope Metal path.");
            return 0;
        }

        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding = create_standard_binding(context.value, result_desc);
        TensorBinding src_binding = create_standard_binding(context.value, src_desc);
        ggml_tensor* position_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I32, flat_row_count(src_desc));
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            position_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        ggml_tensor* contiguous_result = add_to_result ? ggml_cont(context.value, result_binding.tensor) : nullptr;
        if (contiguous_src == nullptr || (add_to_result && contiguous_result == nullptr))
        {
            set_last_error("Failed to create ggml contiguous rope inputs.");
            return 0;
        }

        const std::int64_t rows = flat_row_count(src_desc);
        ggml_tensor* rope_input = ggml_reshape_4d(context.value, contiguous_src, src_desc.ne0, 1, rows, 1);
        ggml_tensor* rope_tensor = nullptr;
        bool use_native_backward = false;
        if (rope_input != nullptr && invert_positions)
        {
            ggml_tensor* native_backward = ggml_rope_ext_back(
                context.value,
                rope_input,
                position_tensor,
                nullptr,
                src_desc.ne0,
                0,
                0,
                500000.0f,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f);
            if (backend_supports_op(native_backward))
            {
                rope_tensor = native_backward;
                use_native_backward = true;
            }
        }

        if (rope_tensor == nullptr)
        {
            rope_tensor = rope_input == nullptr ? nullptr : ggml_rope_ext(
                context.value,
                rope_input,
                position_tensor,
                nullptr,
                src_desc.ne0,
                0,
                0,
                500000.0f,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f);
        }
        ggml_tensor* restored = rope_tensor == nullptr ? nullptr : ggml_reshape_4d(context.value, rope_tensor, result_desc.ne0, result_desc.ne1, result_desc.ne2, result_desc.ne3);
        ggml_tensor* value_tensor = restored;
        if (add_to_result)
        {
            value_tensor = restored == nullptr ? nullptr : ggml_add(context.value, contiguous_result, restored);
        }

        if (rope_input == nullptr || rope_tensor == nullptr || value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml rope node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml rope output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);

        std::vector<std::int32_t> positions(static_cast<std::size_t>(rows));
        for (std::size_t i = 0; i < positions.size(); ++i)
        {
            std::int32_t position = static_cast<std::int32_t>(row_offset + static_cast<int>(i % static_cast<std::size_t>(seq_len)));
            positions[i] = (invert_positions && !use_native_backward) ? -position : position;
        }
        ggml_backend_tensor_set(position_tensor, positions.data(), 0, positions.size() * sizeof(std::int32_t));

        if (add_to_result || result_binding.raw_bytes > logical_bytes(result_desc))
        {
            upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int softmax_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Source tensor shape does not match result shape for ggml softmax.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml softmax Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous softmax input.");
            return 0;
        }

        ggml_tensor* softmax_tensor = ggml_soft_max(context.value, contiguous_src);
        if (softmax_tensor == nullptr)
        {
            set_last_error("Failed to create ggml softmax node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, softmax_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int softmax_grad_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& adj_desc,
        const TensorView4DDesc& val_desc,
        bool add_grad)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(adj_desc, "adj") || !validate_desc(val_desc, "val"))
        {
            return 0;
        }

        if (!same_shape(result_desc, adj_desc) || !same_shape(result_desc, val_desc))
        {
            set_last_error("Tensor shape mismatch passed to ggml softmaxgrad.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(adj_desc) || !can_map_standard_view(val_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml softmaxgrad Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(adj_desc) && can_map_standard_view(val_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding adj_binding;
        TensorBinding val_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, adj_desc, adj_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, val_desc, val_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            adj_binding = create_standard_binding(context.value, adj_desc);
            val_binding = create_standard_binding(context.value, val_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            adj_binding.storage == nullptr || adj_binding.tensor == nullptr ||
            val_binding.storage == nullptr || val_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_adj = ggml_cont(context.value, adj_binding.tensor);
        ggml_tensor* contiguous_val = ggml_cont(context.value, val_binding.tensor);
        if (contiguous_adj == nullptr || contiguous_val == nullptr)
        {
            set_last_error("Failed to create ggml contiguous softmaxgrad inputs.");
            return 0;
        }

        ggml_tensor* grad_tensor = ggml_soft_max_ext_back(context.value, contiguous_adj, contiguous_val, 1.0f, 0.0f);
        if (!backend_supports_op(grad_tensor))
        {
            ggml_tensor* weighted_adj = ggml_mul(context.value, contiguous_val, contiguous_adj);
            if (weighted_adj == nullptr)
            {
                set_last_error("Failed to create ggml softmaxgrad mul node.");
                return 0;
            }

            ggml_tensor* row_sum = ggml_sum_rows(context.value, weighted_adj);
            if (row_sum == nullptr)
            {
                set_last_error("Failed to create ggml softmaxgrad sum_rows node.");
                return 0;
            }

            ggml_tensor* centered_adj = ggml_sub(context.value, contiguous_adj, row_sum);
            if (centered_adj == nullptr)
            {
                set_last_error("Failed to create ggml softmaxgrad subtract node.");
                return 0;
            }

            grad_tensor = ggml_mul(context.value, contiguous_val, centered_adj);
        }

        if (grad_tensor == nullptr)
        {
            set_last_error("Failed to create ggml softmaxgrad output node.");
            return 0;
        }

        if (add_grad)
        {
            ggml_tensor* contiguous_result = ggml_cont(context.value, result_binding.tensor);
            if (contiguous_result == nullptr)
            {
                set_last_error("Failed to create ggml contiguous softmaxgrad accumulation input.");
                return 0;
            }

            grad_tensor = ggml_add(context.value, grad_tensor, contiguous_result);
            if (grad_tensor == nullptr)
            {
                set_last_error("Failed to create ggml softmaxgrad accumulation node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, grad_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml softmaxgrad output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(adj_binding, adj_desc.data, adj_binding.raw_bytes);
            upload_binding(val_binding, val_desc.data, val_binding.raw_bytes);
            if (add_grad || result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int cross_entropy_loss_f32_impl(
        float* loss_value,
        const TensorView4DDesc& probs_desc,
        const ContiguousTensorDesc& target_indices_desc,
        float smooth,
        float label_smooth)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (loss_value == nullptr)
        {
            set_last_error("Null pointer passed for lossValue.");
            return 0;
        }

        if (!validate_desc(probs_desc, "probs") || !validate_desc(target_indices_desc, "targetIndices"))
        {
            return 0;
        }

        if (!can_map_standard_view(probs_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml crossentropyloss Metal path.");
            return 0;
        }

        if (label_smooth < 0.0f || label_smooth > 1.0f)
        {
            set_last_error("labelSmooth must be in [0, 1] for ggml crossentropyloss.");
            return 0;
        }

        const std::int64_t rows = flat_row_count(probs_desc);
        const std::int64_t cols = probs_desc.ne0;
        if (target_indices_desc.element_count != rows)
        {
            set_last_error("Target index count must match the number of probability rows for ggml crossentropyloss.");
            return 0;
        }

        bool probs_zero_copy = can_map_standard_view(probs_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding probs_binding;
        if (probs_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, probs_desc, probs_binding, buf))
                probs_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!probs_zero_copy)
            probs_binding = create_standard_binding(context.value, probs_desc);
        TensorBinding labels_binding = create_matrix_binding(context.value, cols, rows);

        if (probs_binding.storage == nullptr || probs_binding.tensor == nullptr ||
            labels_binding.storage == nullptr || labels_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors for crossentropyloss.");
            return 0;
        }

        ggml_tensor* contiguous_probs = ggml_cont(context.value, probs_binding.tensor);
        ggml_tensor* flat_probs = contiguous_probs == nullptr ? nullptr : flatten_to_rows(context.value, contiguous_probs, cols, rows);
        ggml_tensor* logits_tensor = flat_probs == nullptr ? nullptr : ggml_log(context.value, flat_probs);
        if (contiguous_probs == nullptr || flat_probs == nullptr || logits_tensor == nullptr)
        {
            set_last_error("Failed to create ggml crossentropyloss logits tensor.");
            return 0;
        }

        ggml_tensor* loss_tensor = ggml_cross_entropy_loss(context.value, logits_tensor, labels_binding.tensor);
        if (loss_tensor == nullptr)
        {
            set_last_error("Failed to create ggml_cross_entropy_loss node.");
            return 0;
        }

        if (!backend_supports_op(loss_tensor))
        {
            set_last_error("ggml_cross_entropy_loss is not supported by the active backend.");
            return 0;
        }

        ggml_set_output(loss_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, loss_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        std::vector<float> labels;
        if (!build_cross_entropy_label_buffer(labels, target_indices_desc, rows, cols, label_smooth))
        {
            return 0;
        }

        if (!probs_zero_copy)
            upload_binding(probs_binding, probs_desc.data, probs_binding.raw_bytes);
        upload_binding(labels_binding, labels.data(), labels_binding.raw_bytes);

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(loss_tensor, loss_value, 0, sizeof(float));

        clear_last_error();
        return 1;
    }

    int cross_entropy_loss_backward_f32_impl(
        const TensorView4DDesc& grad_desc,
        const TensorView4DDesc& probs_desc,
        const ContiguousTensorDesc& target_indices_desc,
        float loss_gradient,
        float smooth,
        float label_smooth,
        bool add_grad)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(grad_desc, "grad") || !validate_desc(probs_desc, "probs") || !validate_desc(target_indices_desc, "targetIndices"))
        {
            return 0;
        }

        if (!same_shape(grad_desc, probs_desc))
        {
            set_last_error("Gradient tensor shape must match probability tensor shape for ggml crossentropyloss backward.");
            return 0;
        }

        if (!can_map_standard_view(grad_desc) || !can_map_standard_view(probs_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml crossentropyloss backward Metal path.");
            return 0;
        }

        if (label_smooth < 0.0f || label_smooth > 1.0f)
        {
            set_last_error("labelSmooth must be in [0, 1] for ggml crossentropyloss backward.");
            return 0;
        }

        const std::int64_t rows = flat_row_count(probs_desc);
        const std::int64_t cols = probs_desc.ne0;
        if (target_indices_desc.element_count != rows)
        {
            set_last_error("Target index count must match the number of probability rows for ggml crossentropyloss backward.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(grad_desc) && can_map_standard_view(probs_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 6 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding grad_binding;
        TensorBinding probs_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, grad_desc, grad_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, probs_desc, probs_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            grad_binding = create_standard_binding(context.value, grad_desc);
            probs_binding = create_standard_binding(context.value, probs_desc);
        }
        TensorBinding labels_binding = create_matrix_binding(context.value, cols, rows);
        TensorBinding loss_grad_binding = create_scalar_binding(context.value);

        if (grad_binding.storage == nullptr || grad_binding.tensor == nullptr ||
            probs_binding.storage == nullptr || probs_binding.tensor == nullptr ||
            labels_binding.storage == nullptr || labels_binding.tensor == nullptr ||
            loss_grad_binding.storage == nullptr || loss_grad_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors for crossentropyloss backward.");
            return 0;
        }

        ggml_tensor* contiguous_probs = ggml_cont(context.value, probs_binding.tensor);
        ggml_tensor* flat_probs = contiguous_probs == nullptr ? nullptr : flatten_to_rows(context.value, contiguous_probs, cols, rows);
        ggml_tensor* logits_tensor = flat_probs == nullptr ? nullptr : ggml_log(context.value, flat_probs);
        if (contiguous_probs == nullptr || flat_probs == nullptr || logits_tensor == nullptr)
        {
            set_last_error("Failed to create ggml crossentropyloss backward logits tensor.");
            return 0;
        }

        ggml_tensor* grad_tensor = ggml_cross_entropy_loss_back(context.value, loss_grad_binding.tensor, logits_tensor, labels_binding.tensor);
        if (grad_tensor == nullptr)
        {
            set_last_error("Failed to create ggml_cross_entropy_loss_back node.");
            return 0;
        }

        if (!backend_supports_op(grad_tensor))
        {
            set_last_error("ggml_cross_entropy_loss_back is not supported by the active backend.");
            return 0;
        }

        ggml_tensor* reshaped_grad = ggml_reshape_4d(context.value, grad_tensor, grad_desc.ne0, grad_desc.ne1, grad_desc.ne2, grad_desc.ne3);
        if (reshaped_grad == nullptr)
        {
            set_last_error("Failed to reshape ggml crossentropyloss backward tensor.");
            return 0;
        }

        if (add_grad)
        {
            ggml_tensor* contiguous_grad = ggml_cont(context.value, grad_binding.tensor);
            reshaped_grad = contiguous_grad == nullptr ? nullptr : ggml_add(context.value, contiguous_grad, reshaped_grad);
            if (reshaped_grad == nullptr)
            {
                set_last_error("Failed to create ggml crossentropyloss backward accumulation node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, reshaped_grad, grad_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml crossentropyloss backward output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        std::vector<float> labels;
        if (!build_cross_entropy_label_buffer(labels, target_indices_desc, rows, cols, label_smooth))
        {
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(probs_binding, probs_desc.data, probs_binding.raw_bytes);
            if (add_grad || grad_binding.raw_bytes > logical_bytes(grad_desc))
                upload_binding(grad_binding, grad_desc.data, grad_binding.raw_bytes);
        }
        upload_binding(labels_binding, labels.data(), labels_binding.raw_bytes);
        ggml_backend_tensor_set(loss_grad_binding.storage, &loss_gradient, 0, sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(grad_binding.storage, grad_desc.data, 0, grad_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int adam_f32_impl(
        const ContiguousTensorDesc& weight_desc,
        const ContiguousTensorDesc& gradient_desc,
        const ContiguousTensorDesc& v_desc,
        const ContiguousTensorDesc& m_desc,
        float grad_norm_factor,
        float step_size,
        float clip_value,
        float regc,
        float decay_rate_v,
        float decay_rate_m,
        int iter,
        float eps)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(weight_desc, "weight")
            || !validate_desc(gradient_desc, "gradient")
            || !validate_desc(v_desc, "v")
            || !validate_desc(m_desc, "m"))
        {
            return 0;
        }

        if (weight_desc.element_count != gradient_desc.element_count
            || weight_desc.element_count != v_desc.element_count
            || weight_desc.element_count != m_desc.element_count)
        {
            set_last_error("Tensor shape mismatch passed to ggml adam.");
            return 0;
        }

        bool use_zero_copy = true;
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding weight_binding;
        TensorBinding gradient_binding;
        TensorBinding v_binding;
        TensorBinding m_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_contiguous(context.value, g_backend, weight_desc, weight_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_contiguous(context.value, g_backend, gradient_desc, gradient_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_contiguous(context.value, g_backend, v_desc, v_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_contiguous(context.value, g_backend, m_desc, m_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            weight_binding = create_contiguous_binding(context.value, weight_desc);
            gradient_binding = create_contiguous_binding(context.value, gradient_desc);
            v_binding = create_contiguous_binding(context.value, v_desc);
            m_binding = create_contiguous_binding(context.value, m_desc);
        }
        ggml_tensor* adamw_params_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, 7);

        if (weight_binding.storage == nullptr || weight_binding.tensor == nullptr ||
            gradient_binding.storage == nullptr || gradient_binding.tensor == nullptr ||
            v_binding.storage == nullptr || v_binding.tensor == nullptr ||
            m_binding.storage == nullptr || m_binding.tensor == nullptr ||
            adamw_params_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors for adam.");
            return 0;
        }

        ggml_tensor* grad_tensor = gradient_binding.tensor;
        if (grad_norm_factor != 1.0f)
        {
            grad_tensor = ggml_scale(context.value, grad_tensor, grad_norm_factor);
            if (grad_tensor == nullptr)
            {
                set_last_error("Failed to create ggml adam grad scaling node.");
                return 0;
            }
        }

        ggml_tensor* clipped_grad = ggml_clamp(context.value, grad_tensor, -clip_value, clip_value);
        if (clipped_grad == nullptr)
        {
            set_last_error("Failed to create ggml adam clamp node.");
            return 0;
        }

        const float bias_correction_m = static_cast<float>(1.0 / (1.0 - std::pow(decay_rate_m, iter)));
        const float bias_correction_v = static_cast<float>(1.0 / (1.0 - std::pow(decay_rate_v, iter)));
        const std::array<float, 7> adamw_params = {
            step_size,
            decay_rate_m,
            decay_rate_v,
            eps,
            regc,
            bias_correction_m,
            bias_correction_v
        };

        ggml_set_param(weight_binding.tensor);

        ggml_tensor* adamw_step = ggml_opt_step_adamw(
            context.value,
            weight_binding.tensor,
            clipped_grad,
            m_binding.tensor,
            v_binding.tensor,
            adamw_params_tensor);
        if (adamw_step == nullptr)
        {
            set_last_error("Failed to create ggml adamw optimizer node.");
            return 0;
        }

        ggml_set_output(adamw_step);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, adamw_step);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(weight_binding, weight_desc.data, weight_binding.raw_bytes);
            upload_binding(gradient_binding, gradient_desc.data, gradient_binding.raw_bytes);
            upload_binding(v_binding, v_desc.data, v_binding.raw_bytes);
            upload_binding(m_binding, m_desc.data, m_binding.raw_bytes);
        }
        ggml_backend_tensor_set(adamw_params_tensor, adamw_params.data(), 0, adamw_params.size() * sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_memset(gradient_binding.storage, 0, 0, gradient_binding.raw_bytes);
        ggml_backend_synchronize(g_backend);

        if (!use_zero_copy)
        {
            ggml_backend_tensor_get(weight_binding.storage, weight_desc.data, 0, weight_binding.raw_bytes);
            ggml_backend_tensor_get(m_binding.storage, m_desc.data, 0, m_binding.raw_bytes);
            ggml_backend_tensor_get(v_binding.storage, v_desc.data, 0, v_binding.raw_bytes);
            ggml_backend_tensor_get(gradient_binding.storage, gradient_desc.data, 0, gradient_binding.raw_bytes);
        }

        clear_last_error();
        return 1;
    }
}

TSG_EXPORT const char* TSGgml_GetLastError()
{
    return g_last_error.c_str();
}

TSG_EXPORT int TSGgml_IsMetalAvailable()
{
    clear_last_error();
    return ensure_backend(BACKEND_TYPE_METAL) ? 1 : 0;
}

TSG_EXPORT int TSGgml_IsBackendAvailable(int backendType)
{
    clear_last_error();
    return ensure_backend(backendType) ? 1 : 0;
}

TSG_EXPORT int TSGgml_AddmmF32(
    TensorView2DDesc result,
    TensorView2DDesc src,
    TensorView2DDesc m1,
    TensorView2DDesc m2,
    float beta,
    float alpha)
{
    try
    {
        return addmm_f32_impl(result, src, m1, m2, beta, alpha);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml addmm failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_AddmmBatchF32(
    TensorView3DDesc result,
    TensorView3DDesc src,
    TensorView3DDesc m1,
    TensorView3DDesc m2,
    float beta,
    float alpha)
{
    try
    {
        return addmm_batch_f32_impl(result, src, m1, m2, beta, alpha);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml addmmbatch failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_ReduceLastDimF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src)
{
    try
    {
        return reduce_last_dim_f32_impl(static_cast<ReductionOpCode>(op), result, src);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml reduction failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_IndexReductionF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src)
{
    try
    {
        return index_reduction_f32_impl(static_cast<IndexReductionOpCode>(op), result, src);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml index-reduction failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_CopyF32(
    TensorView4DDesc result,
    TensorView4DDesc src)
{
    try
    {
        return copy_f32_impl(result, src);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml copy failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_UnaryF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src)
{
    try
    {
        return unary_f32_impl(static_cast<UnaryOpCode>(op), result, src);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml unary failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_BinaryTensorF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc lhs,
    TensorView4DDesc rhs)
{
    try
    {
        return binary_tensor_f32_impl(static_cast<BinaryTensorOpCode>(op), result, lhs, rhs);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml binary-tensor failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_BinaryScalarF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src,
    float scalar)
{
    try
    {
        return binary_scalar_f32_impl(static_cast<BinaryScalarOpCode>(op), result, src, scalar);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml binary-scalar failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_ActivationGradF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src,
    TensorView4DDesc grad,
    TensorView4DDesc accumulation,
    int has_accumulation)
{
    try
    {
        return activation_grad_f32_impl(static_cast<ActivationGradOpCode>(op), result, src, grad, accumulation, has_accumulation != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml activation-grad failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_NormF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src,
    TensorView4DDesc gamma,
    TensorView4DDesc beta,
    int has_beta,
    float eps)
{
    try
    {
        return norm_f32_impl(static_cast<NormOpCode>(op), result, src, gamma, beta, has_beta != 0, eps);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml norm failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_NormGradF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc grad_gamma,
    TensorView4DDesc grad_beta,
    TensorView4DDesc adj,
    TensorView4DDesc x,
    TensorView4DDesc gamma,
    int has_grad_beta,
    float eps)
{
    try
    {
        return norm_grad_f32_impl(static_cast<NormOpCode>(op), result, grad_gamma, grad_beta, adj, x, gamma, has_grad_beta != 0, eps);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml norm-grad failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_IndexSelectF32(
    TensorView2DDesc result,
    TensorView2DDesc src,
    ContiguousTensorDesc indices,
    int add_to_result)
{
    try
    {
        return index_select_f32_impl(result, src, indices, add_to_result != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml indexselect failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_IndexSelectGradF32(
    TensorView2DDesc grad,
    TensorView2DDesc adj,
    ContiguousTensorDesc indices)
{
    try
    {
        return index_select_grad_f32_impl(grad, adj, indices);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml indexselectgrad failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_RoPEF32(
    TensorView4DDesc result,
    TensorView4DDesc src,
    int seq_len,
    int row_offset,
    int add_to_result,
    int invert_positions)
{
    try
    {
        return rope_f32_impl(result, src, seq_len, row_offset, add_to_result != 0, invert_positions != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml rope failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_SoftmaxF32(
    TensorView4DDesc result,
    TensorView4DDesc src)
{
    try
    {
        return softmax_f32_impl(result, src);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml softmax failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_SoftmaxGradF32(
    TensorView4DDesc result,
    TensorView4DDesc adj,
    TensorView4DDesc val,
    int add_grad)
{
    try
    {
        return softmax_grad_f32_impl(result, adj, val, add_grad != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml softmaxgrad failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_CrossEntropyLossF32(
    float* loss_value,
    TensorView4DDesc probs,
    ContiguousTensorDesc target_indices,
    float smooth,
    float label_smooth)
{
    try
    {
        return cross_entropy_loss_f32_impl(loss_value, probs, target_indices, smooth, label_smooth);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml crossentropyloss failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_CrossEntropyLossBackwardF32(
    TensorView4DDesc grad,
    TensorView4DDesc probs,
    ContiguousTensorDesc target_indices,
    float loss_gradient,
    float smooth,
    float label_smooth,
    int add_grad)
{
    try
    {
        return cross_entropy_loss_backward_f32_impl(grad, probs, target_indices, loss_gradient, smooth, label_smooth, add_grad != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml crossentropyloss backward failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_AdamF32(
    ContiguousTensorDesc weight,
    ContiguousTensorDesc gradient,
    ContiguousTensorDesc v,
    ContiguousTensorDesc m,
    float grad_norm_factor,
    float step_size,
    float clip_value,
    float regc,
    float decay_rate_v,
    float decay_rate_m,
    int iter,
    float eps)
{
    try
    {
        return adam_f32_impl(weight, gradient, v, m, grad_norm_factor, step_size, clip_value, regc, decay_rate_v, decay_rate_m, iter, eps);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml adam failure.");
        return 0;
    }
}
