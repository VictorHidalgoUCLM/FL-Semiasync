import configparser

# Crear el objeto ConfigParser
config = configparser.ConfigParser()

# Configurar las secciones y los valores
config['configDevices'] = {
    'raspberrypi1': "['raspberrypi1', 'ES', 'RP1', 'victor']",
    'raspberry4': "['raspberry4', 'ES', 'RP4', 'victor']",
    'raspberry3': "['raspberry3', 'ES', 'RP3', 'victor']",
    'raspberry6': "['raspberry6', 'ES', 'RP6', 'victor']",
    'raspberry7': "['raspberry7', 'ES', 'RP7', 'victor']",
}

# Escribir el archivo de configuración
with open('config.ini', 'w') as configfile:
    config.write(configfile)

print("Archivo config.ini generado con éxito.")
