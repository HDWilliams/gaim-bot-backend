# gaim-bot-backend
## Backend for updated gaim-bot v2 chatbot
## https://gaimbotv2.streamlit.app/

## Project Overview
Gaim-bot is a retrieval and AI chatbot to answer natural language queries about the video game eldin ring. This backend repository contains funcationality for an AWS lambda function that
1. Accepts api requests authenticated with an api key
2. Performs retieval on a Pinecone vector store to obtain relevant context
3. Integrates that context with the user query and uses the OpenAI api to get a generated response to return in the api response

While this project is uses information related to Eldin Ring, this backend RAG pipeline is information agnostic and can be extended to use other foundation models and additional retrieval options such as FAISS.

## Project Architecture
Overview: 
The core of this project was bulding out a serverless scalable retrieval augmented generation pipeline that could utilize external knowledge sources and foundation model apis to answer user queries. AWS Lambda was chosen as it automatically scales and is highly cost effective. However it is also stateless, so the state needed to process the user query and the retrieval occur in modules outside of the lambda function. The client request contains the user query, message history and the instructions for the LLM. For retrieval Pinecone, a hosted vector store, was selected for simplicity and to avoid the cold start time of loading a FAISS index into memory. Embedding is also done through the Open AI api given the resource constraints of a Lambda function vs a dedicated EC2 server. 

Finally, document uploading is currently done with a locally running function provided in the repo, however this can easily be extended to its own lambda function that can process documents uploaded to S3 into a Pinecone index.  

Below is an overview of all services used for this project
- AWS API Gateway: creates and api endpoint
- AWS Lambda: serverless RAG function execution
- AWS S3: used as storage for a Lambda Layer (stores python packages)
- Pinecone: hosted vector store for document retrieval 
- OpenAI API
  

## Set Up and Running the Project
It is recommended to start with setting up the necessary infrastructre and testing to ensure all the necessary pieces are connected properly
1. Create an AWS account and create a Lambda function
2. Connect with Lambda function to API gateway to expose an api endpoint; at this point test the endpoint locally ex/ Postman
3. Create a Pinecone (https://www.pinecone.io/) account and create a vector store; test it locally. This project uses Langchain to interact with Pinecone, an individual functions in PineconeInterface.py can be run locally for testing.

At this point we will need to create a lambda layer with our necessary packages for this project. AWS runs various versions of Linux. If you are on a Linux machine with python 3.12, you can simply
1. git clone git@github.com:HDWilliams/gaim-bot-backend.git
2. start a virtual environment
3. pip install -r requirements.txt
4. copy the following file path and all contained files into a zipped folder 'python\lib\python3.12\site-packages'. Upload this folder to S3 and use it to create a lambda layer

If you are not using a Linux machine, follow these steps. Note: docker is required
1. git clone git@github.com:HDWilliams/gaim-bot-backend.git
2. copy the 'requirements.txt' file from this project into another folder
3. make the following directory structure 'python\lib\python3.12\site-packages' `mkdir -p python/lib/python3.12/site-packages`
4. then `pip install -r /path/to/requirements.txt --platform manylinux2014_x86_64 --target . --implementation cp --only-binary=:all:` to pip install the Linux version of the needed packages
5. zip the 'python' folder to include the packages and entire folder structure. Upload this folder to S3 and use it to create a lambda layer

Finally add all the file in the 'chat_response_lambda' folder to the lambda function file system, replacing the existing lambda_function.py

  

