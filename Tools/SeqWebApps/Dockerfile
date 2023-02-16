# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb
RUN rm packages-microsoft-prod.deb

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh

RUN apt-get update
RUN apt-get install -y dotnet-sdk-7.0
RUN apt-get install -y aspnetcore-runtime-7.0
RUN apt-get install -y cmake
RUN apt-get install -y git-lfs

RUN git clone https://github.com/zhongkaifu/Seq2SeqSharp.git
WORKDIR /code/Seq2SeqSharp
RUN dotnet build Seq2SeqSharp.sln --configuration Release

WORKDIR /code/Seq2SeqSharp/ExternalProjects
RUN unzip SentencePiece.zip
WORKDIR /code/Seq2SeqSharp/ExternalProjects/SentencePiece
RUN mkdir build
WORKDIR /code/Seq2SeqSharp/ExternalProjects/SentencePiece/build
RUN cmake ..
RUN make -j $(nproc)
RUN make install
RUN ldconfig -v

WORKDIR /code

#RUN git clone https://huggingface.co/zhongkaifu/mt_enu_chs

RUN mkdir -p /code/bin
RUN chmod 777 /code/bin
WORKDIR /code/bin

RUN cp -r /code/Seq2SeqSharp/Tools/SeqWebApps/bin/Release/net7.0/* .
RUN wget https://github.com/zhongkaifu/Models/releases/download/MT_ENU_CHS/mt_enu_chs.model
RUN wget https://huggingface.co/zhongkaifu/mt_enu_chs/resolve/main/enuSpm.model
RUN wget https://huggingface.co/zhongkaifu/mt_enu_chs/resolve/main/cjkSpm.model
RUN rm appsettings.json
RUN wget https://huggingface.co/zhongkaifu/mt_enu_chs/resolve/main/appsettings.json
#RUN cp /code/mt_enu_chs/appsettings.json .

CMD ["dotnet","/code/bin/SeqWebApps.dll"]
