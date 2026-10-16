# Datos

Toda la carpeta `data/` se versiona (por ahora pesa ~67 MB; el archivo más grande es `raw/gtfs/stop_times.txt`, ~49 MiB).
Los CSV de `raw/` son los datos de confianza del proyecto.

- `raw/` — datos originales sin modificar: GTFS, depósitos y cargadores, tarifas (`tariffs/electricity_prices.csv`), flota, y parámetros globales con su glosario (`other/`).
- `interim/` — datos derivados intermedios (`shape_distances.csv`, largo de cada trazado; `deadhead_VN-10.csv`, matriz de deadheads estimada).
- `processed/<particion>/` — salida de la partición (viajes, plan de bloques, saturación de terminales, Excel piloto). Se regeneran con `python scripts/generar_subproblemas.py`.

Formatos esperados en [../docs/datos/datos_requeridos.md](../docs/datos/datos_requeridos.md).
La partición está descrita en [decisiones/instancias.md](../docs/decisiones/instancias.md).
