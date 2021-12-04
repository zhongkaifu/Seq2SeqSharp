using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ProtoBuf;

using AdvUtils;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Models
{
    /// <summary>
    /// 
    /// </summary>
    [ProtoContract(SkipConstructor=true)]
    public sealed class Vocab_4_ProtoBufSerializer
    {
        public const int START_MEANING_INDEX = 3;

        [ProtoMember(1)] private Dictionary<string, int> _WordToIndex;
        [ProtoMember(2)] private Dictionary<int, string> _IndexToWord;
        [ProtoMember(3)] private bool _IgnoreCase;
        public int Count => _IndexToWord.Count;
        public bool IgnoreCase => _IgnoreCase;
        public IReadOnlyCollection<string> Items => _WordToIndex.Keys;
        public Dictionary<string, int> _GetWordToIndex_() => _WordToIndex;
        public Dictionary<int, string> _GetIndexToWord_() => _IndexToWord;
        public Vocab ToVocab() => new Vocab( this );

        public static (Dictionary<string, int> wordToIndex, Dictionary<int, string> indexToWord, bool ignoreCase) CreateDicts( bool ignoreCase )
        {
            var wordToIndex = ignoreCase ? new Dictionary<string, int>( StringComparer.InvariantCultureIgnoreCase ) : new Dictionary<string, int>();
            var indexToWord = new Dictionary<int, string>();

            wordToIndex[ BuildInTokens.EOS ] = (int) SENTTAGS.END;
            wordToIndex[ BuildInTokens.BOS ] = (int) SENTTAGS.START;
            wordToIndex[ BuildInTokens.UNK ] = (int) SENTTAGS.UNK;

            indexToWord[ (int) SENTTAGS.END ] = BuildInTokens.EOS;
            indexToWord[ (int) SENTTAGS.START ] = BuildInTokens.BOS;
            indexToWord[ (int) SENTTAGS.UNK ] = BuildInTokens.UNK;

            return (wordToIndex, indexToWord, ignoreCase);
        }

        public Vocab_4_ProtoBufSerializer( Vocab v ) => (_WordToIndex, _IndexToWord, _IgnoreCase) = (v._GetWordToIndex_(), v._GetIndexToWord_(), v.IgnoreCase);
        public Vocab_4_ProtoBufSerializer( bool ignoreCase ) => (_WordToIndex, _IndexToWord, _IgnoreCase) = CreateDicts( ignoreCase );
        public Vocab_4_ProtoBufSerializer( Dictionary<string, int> wordToIndex, Dictionary<int, string> indexToWord ) => (_WordToIndex, _IndexToWord) = (wordToIndex, indexToWord);
        public Vocab_4_ProtoBufSerializer( in (Dictionary<string, int> wordToIndex, Dictionary<int, string> indexToWord) t ) => (_WordToIndex, _IndexToWord) = t;
        /// <summary>
        /// Load vocabulary from given files
        /// </summary>
        public Vocab_4_ProtoBufSerializer( string vocabFilePath, bool ignoreCase )
        {
            Logger.WriteLine( "Loading vocabulary files..." );

            (_WordToIndex, _IndexToWord, _IgnoreCase) = CreateDicts( ignoreCase );

            using var sr = new StreamReader( vocabFilePath );
            //Build word index for both source and target sides
            int q = START_MEANING_INDEX;
            for ( var line = sr.ReadLine(); line != null; line = sr.ReadLine() )
            {
                var idx = line.IndexOf( '\t' );
                var word = (idx == -1) ? line : line.Substring( 0, idx );
                if (  word.IsNullOrEmpty() ) continue;

                if ( !BuildInTokens.IsPreDefinedToken( word ) )
                {
                    _WordToIndex[ word ] = q;
                    _IndexToWord[ q ] = word;
                    q++;
                }
            }
        }

        public void DumpVocab( string fileName )
        {
            using var sw = new StreamWriter( fileName );
            foreach ( KeyValuePair<int, string> pair in _IndexToWord )
            {
                sw.Write( pair.Value );
                sw.Write( '\t' );
                sw.WriteLine( pair.Key );
            }
        }

        public string GetString( int idx )
        {
            if ( _IndexToWord.TryGetValue( idx, out var letter ) )
            {
                return (letter);
            }
            return (BuildInTokens.UNK);
        }
        public List<string> ConvertIdsToString( IList<float> idxs )
        {
            var result = new List<string>( idxs.Count );
            foreach ( int idx in idxs )
            {
                if ( !_IndexToWord.TryGetValue( idx, out var letter ) )
                {
                    letter = BuildInTokens.UNK;
                }
                result.Add( letter );
            }
            return (result);
        }
        public string ConvertIdsToString( int idx )
        {
            if ( !_IndexToWord.TryGetValue( idx, out var letter ) )
            {
                letter = BuildInTokens.UNK;
            }
            return (letter);
        }

        public List<List<string>> ConvertIdsToString( List<List<int>> seqs )
        {
            var result = new List<List<string>>( seqs.Count );
            foreach ( var seq in seqs )
            {
                var r = new List<string>( seq.Count );
                foreach ( int idx in seq )
                {
                    if ( !_IndexToWord.TryGetValue( idx, out string letter ) )
                    {
                        letter = BuildInTokens.UNK;
                    }
                    r.Add( letter );
                }

                result.Add( r );
            }
            return (result);
        }
        public List<List<List<string>>> ConvertIdsToString( List<List<List<int>>> beam2seqs )
        {
            var result = new List<List<List<string>>>( beam2seqs.Count );
            foreach ( var seqs in beam2seqs )
            {
                var b = new List<List<string>>( seqs.Count );
                foreach ( var seq in seqs )
                {
                    var r = new List<string>( seq.Count );
                    foreach ( int idx in seq )
                    {
                        if ( !_IndexToWord.TryGetValue( idx, out string letter ) )
                        {
                            letter = BuildInTokens.UNK;
                        }
                        r.Add( letter );
                    }

                    b.Add( r );
                }
                result.Add( b );
            }
            return (result);
        }

        public bool ContainsWord( string word ) => _WordToIndex.ContainsKey( word );
        public int GetWordIndex( string word, bool logUnk = false )
        {
            if ( !_WordToIndex.TryGetValue( word, out int id ) )
            {
                id = (int) SENTTAGS.UNK;
                if ( logUnk )
                {
                    Logger.WriteLine( $"Source word '{word}' is UNK" );
                }
            }
            return (id);
        }

        public List<List<int>> GetWordIndex( List<IList<string>> seqs, bool logUnk = false )
        {
            var result = new List<List<int>>( seqs.Count );
            foreach ( var seq in seqs )
            {
                var r = new List<int>( seq.Count );
                foreach ( var word in seq )
                {
                    if ( !_WordToIndex.TryGetValue( word, out int id ) )
                    {
                        id = (int) SENTTAGS.UNK;
                        if ( logUnk )
                        {
                            Logger.WriteLine( $"Source word '{word}' is UNK" );
                        }
                    }
                    r.Add( id );
                }
                result.Add( r );
            }
            return (result);
        }
        public List<List<int>> GetWordIndex( List<List<string>> seqs, bool logUnk = false )
        {
            var result = new List<List<int>>( seqs.Count );
            foreach ( var seq in seqs )
            {
                var r = new List<int>( seq.Count );
                foreach ( var word in seq )
                {
                    if ( !_WordToIndex.TryGetValue( word, out int id ) )
                    {
                        id = (int) SENTTAGS.UNK;
                        if ( logUnk )
                        {
                            Logger.WriteLine( $"Source word '{word}' is UNK" );
                        }
                    }
                    r.Add( id );
                }
                result.Add( r );
            }
            return (result);
        }
    }

    /// <summary>
    /// 
    /// </summary>
    [ProtoContract(SkipConstructor=true), ProtoInclude(100, typeof(Vocab_4_ProtoBufSerializer)),
                                          ProtoInclude(101, typeof(DecoderTypeEnums)),
                                          ProtoInclude(101, typeof(EncoderTypeEnums))/*,
                                          ProtoInclude(102, typeof(SimilarityTypeEnums))*/]
    public sealed class Model_4_ProtoBufSerializer //: IModel
    {
        public Model_4_ProtoBufSerializer() { }
        public Model_4_ProtoBufSerializer( Model m )
        {
            Name2Weights            = m.Name2Weights;
            DecoderEmbeddingDim     = m.DecoderEmbeddingDim;
            EncoderEmbeddingDim     = m.EncoderEmbeddingDim;
            DecoderLayerDepth       = m.DecoderLayerDepth;
            EncoderLayerDepth       = m.EncoderLayerDepth;
            DecoderType             = m.DecoderType;
            EncoderType             = m.EncoderType;
            HiddenDim               = m.HiddenDim;
            EnableSegmentEmbeddings = m.EnableSegmentEmbeddings;
            MultiHeadNum            = m.MultiHeadNum;
            SrcVocab                = new Vocab_4_ProtoBufSerializer( m.SrcVocab );
            TgtVocab                = m.TgtVocab != null ? new Vocab_4_ProtoBufSerializer( m.TgtVocab ) : null;
            ClsVocabs               = m.ClsVocabs?.Select( c => new Vocab_4_ProtoBufSerializer( c ) ).ToList();
            EnableCoverageModel     = m.EnableCoverageModel;
            SharedEmbeddings        = m.SharedEmbeddings;
            //SimilarityType_         = m.SimilarityType;
            SimilarityType          = m.SimilarityType;
            ApplyContextEmbeddingsToEntireSequence
                                    = m.ApplyContextEmbeddingsToEntireSequence;
            MaxSegmentNum           = m.MaxSegmentNum;
            PointerGenerator        = m.PointerGenerator;
        }
        public static Model_4_ProtoBufSerializer Create( Model m ) => new Model_4_ProtoBufSerializer( m );

        [ProtoMember(1)] public Dictionary<string, float[]> Name2Weights { get; set; }
        [ProtoMember(2)] public int DecoderEmbeddingDim { get; set; }
        [ProtoMember(3)] public int EncoderEmbeddingDim { get; set; }
        [ProtoMember(4)] public int DecoderLayerDepth { get; set; }
        [ProtoMember(5)] public int EncoderLayerDepth { get; set; }
        [ProtoMember(6)] public DecoderTypeEnums DecoderType { get; set; }
        [ProtoMember(7)] public EncoderTypeEnums EncoderType { get; set; }
        [ProtoMember(8)] public int HiddenDim { get; set; }
        [ProtoMember(9)] public bool EnableSegmentEmbeddings { get; set; }
        [ProtoMember(10)] public int MultiHeadNum { get; set; }
        //[ProtoMember(11)] public Vocab SrcVocab { get; set; }
        //[ProtoMember(12)] public Vocab TgtVocab { get; set; }
        //[ProtoMember(13)] public List< Vocab > ClsVocabs { get; set; }
        [ProtoMember(11)] public Vocab_4_ProtoBufSerializer SrcVocab { get; set; }
        [ProtoMember(12)] public Vocab_4_ProtoBufSerializer TgtVocab { get; set; }
        [ProtoMember(13)] public List< Vocab_4_ProtoBufSerializer > ClsVocabs { get; set; }
        [ProtoMember(14)] public bool EnableCoverageModel { get; set; }
        [ProtoMember(15)] public bool SharedEmbeddings { get; set; }
        [ProtoMember(16)] public string SimilarityType { get; set; }
        [ProtoMember(17)] public bool ApplyContextEmbeddingsToEntireSequence { get; set; }
        //[ProtoMember(18)] public SimilarityTypeEnums SimilarityType_ { get; set; }
        [ProtoMember(19)] public int MaxSegmentNum { get; set; }

        [ProtoMember(20)] public bool PointerGenerator { get; set; }

        //Vocab IModel.SrcVocab { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        //Vocab IModel.TgtVocab { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        //List<Vocab> IModel.ClsVocabs { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
        //public Vocab ClsVocab { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }
    }
}
