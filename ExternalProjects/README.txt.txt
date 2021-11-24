SentencePiece
SentencePiece is a modified SentencePiece project from Google (https://github.com/google/sentencepiece). Some C# wrapper methods are added to this project.
Run the following commands to unpackage, build and install it:
% unzip SentencePiece.zip
% cd sentencepiece
% mkdir build
% cd build
% cmake ..
% make -j $(nproc)
% sudo make install
% sudo ldconfig -v

