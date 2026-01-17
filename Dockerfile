# Obraz bazowy z Pythonem
FROM python:3.11-slim

# Katalog roboczy
WORKDIR /app

# Kopiuj requirements i instaluj zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
RUN python train.py

# Kopiuj kod aplikacji
COPY app.py .

# Port aplikacji
EXPOSE 5000

# Uruchom aplikację
CMD ["python", "app.py"]
