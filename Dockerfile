FROM dleunji/chatbotmodel
WORKDIR /app
COPY . .
RUN apt-get update && \
    apt-get install -y
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["opyrator", "launch-api", "app:chatbot"]
