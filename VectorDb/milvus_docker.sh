docker stop milvus-standalone
docker rm milvus-standalone
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.2.11 milvus run standalone