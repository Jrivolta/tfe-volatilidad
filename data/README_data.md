# Datos del proyecto

Los precios historicos NO se almacenan en el repositorio. Se descargan en
tiempo de ejecucion mediante la libreria `yfinance`, lo que garantiza la
reproducibilidad sin distribuir datos de terceros.

- Activos: AAPL, AMZN, GOOGL, TSLA (NASDAQ)
- Periodo: 2015-2024
- Frecuencia: diaria (precio de cierre ajustado)
- Variable objetivo: regimen de volatilidad (bajo / medio / alto), etiquetado
  por terciles empiricos de la volatilidad realizada a 21 dias bursatiles.

Para obtener los datos basta con ejecutar el script principal; este descarga,
valida la completitud y construye el conjunto de atributos de forma automatica.
