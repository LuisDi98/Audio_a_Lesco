# Utiliza una imagen base de Python
FROM python:3.8

# Establece el directorio de trabajo en el folder actual
WORKDIR /


# Copia el archivo de requisitos (requirements.txt) al contenedor
COPY requirements.txt .

# Instala las dependencias de la aplicación
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el contenido actual al directorio de trabajo en el contenedor
COPY . .

# Expone el puerto en el que la aplicación Flask se ejecutará
EXPOSE 5000

# Define la variable de entorno para que Flask sepa que se está ejecutando en un contenedor
ENV FLASK_ENV=production

# Ejecuta la aplicación Flask
CMD ["python", "app.py"]
