# "Python 3.8 slim buster" is a Docker image that provides a lightweight and minimalistic environment for running Python 3.8 applications, based on the Debian Buster operating system.
FROM python:3.8-slim

# creating a new  directory 
RUN mkdir /app

# making it as our working directory
WORKDIR /app

# copying everything from current directory to working directory
COPY . /app/

# installing dependencies
RUN pip3 install -r requirements.txt

CMD flask run -h 0.0.0.0 -p 5000


