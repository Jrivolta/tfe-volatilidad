# Datos del proyecto

Los precios historicos NO se almacenan en el repositorio. Se descargan en tiempo de ejecucion mediante la libreria `yfinance`, lo que garantiza la reproducibilidad sin distribuir datos de terceros.

- Activos: AAPL, AMZN, GOOGL, TSLA (NASDAQ)
- Periodo: 2015-2024
- Frecuencia: diaria (precio de cierre ajustado)
- Atributos (10): retornos log rezagados t-1..t-5, SMA 5, SMA 20, RSI 14, volatilidad rezagada t-1 y volumen relativo
- Variable objetivo: regimen de volatilidad (bajo / medio / alto), etiquetado por los terciles empiricos de la volatilidad realizada a 21 dias. Los umbrales de los terciles se estiman unicamente con el tramo de entrenamiento para evitar la filtracion de informacion futura.
- Horizonte configurable (parametro HORIZONTE del script): 0 = regimen del dia actual; 21 = regimen dentro de 21 dias bursatiles.

Para obtener los datos basta con ejecutar el script principal; este descarga, valida la completitud y construye el conjunto de atributos de forma automatica.
