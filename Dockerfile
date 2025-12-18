FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN python model.py

EXPOSE 5000

CMD ["python", "app.py"]
