from loguru import logger
import os

# Carpeta destino del log
path = os.getcwd() + "\\resources\\log"
# Nombre del archivo Log
file_name = 'log'
# Extensión del archivo
file_extension = 'log'

# Ruta completa del nuevo archivo de log
file = f'{path}\\{file_name}.{file_extension}'

# Clase que edita el archivo con los parámetros seleccionados
logger.add(file,  # Ruta donde se guardan los log
           rotation='50 MB',  # Rotation: Rota los log automáticamente una vez que el archivo alcanza el tamaño
           level='TRACE',  # Nivel mínimo desde el cual comienza a guardar log
           retention='5d',  # Tiempo máximo que guarda los log
           # colorize=True,  # Permite añadir colores al log
           enqueue=True,  # Garantiza el envio de logs ya que hace que el envio de los mismos sea asíncrono
           # serialize=True,  # Los registros se convierten en una cadena JSON
           # format="<green>{time}</green> <level>{message}</level>"  # Formato del mensaje
           ) # https://loguru.readthedocs.io/en/stable/api/logger.html
