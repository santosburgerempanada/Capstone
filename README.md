# Cómo correr los experimentos

Instrucciones para reproducir los experimentos del ALNS que ya se corrieron: los días hábiles completos de
Vespucio Norte, El Conquistador y los dos juntos, con las semillas 1, 2 y 3. Comandos para PowerShell, desde la raíz
del repositorio.

## 1. Requisitos

- Python ≥ 3.10.
- Licencia de Gurobi activa (el ALNS usa Gurobi en cada iteración). Comprobar con: `python -c "import gurobipy; gurobipy.Model()"`.
- Instalar el proyecto (una vez):

```powershell
pip install -e ".[dev]"
```

Los datos (`data/`) y las matrices de deadheads (`data/interim/deadhead_*.csv`) vienen en el repositorio.

## 2. Antes de correr

```powershell
git pull origin main
git status
```

`git status` no debe mostrar archivos modificados: cada corrida guarda el commit con que se hizo y si había cambios sin
commitear. `results/` y `logs/` no aparecen porque están en el `.gitignore`.

## 3. Los experimentos

| Instancia | Viajes | Depósitos | Experimento | Commit con que se corrió | Tiempo por corrida¹ | Resultado (semillas 1, 2, 3) |
|---|---|---|---|---|---|---|
| Vespucio Norte (`VN-dia`) | 7 448 | 1 | `E7-vaciar-ajuste` | `429161f` | ~33 min | 783, 784, 784 buses; 228 387 USD en promedio |
| El Conquistador (`EC-dia`) | 17 288 | 1 | `E1-definitiva` | `203259d` | ~1,8 h | 2 168, 2 173, 2 168 buses; 651 585 USD |
| VN + EC juntos (`VN-EC-dia`) | 24 736 | 2 | `E1-definitiva` | `203259d` | ~2,8 h | 2 944, 2 939, 2 946 buses; 875 452 USD |

¹ En un computador de 8 núcleos con 6–9 corridas a la vez; incluye construir el baseline, que cada corrida rehace.

## 4. Correr

Una corrida por semilla:

```powershell
python scripts/run_alns.py --config <archivo de la instancia> --experimento <experimento> --semilla <S>
```

Las nueve corridas, hasta 8 a la vez, con un registro por corrida en `logs\` (no se versiona):

```powershell
$corridas = @(
  "--config config/instances/VN-EC-dia.yaml --experimento E1-definitiva --semilla 1",
  "--config config/instances/VN-EC-dia.yaml --experimento E1-definitiva --semilla 2",
  "--config config/instances/VN-EC-dia.yaml --experimento E1-definitiva --semilla 3",
  "--config config/instances/EC-dia.yaml --experimento E1-definitiva --semilla 1",
  "--config config/instances/EC-dia.yaml --experimento E1-definitiva --semilla 2",
  "--config config/instances/EC-dia.yaml --experimento E1-definitiva --semilla 3",
  "--config config/instances/VN-dia-vaciar-r500.yaml --experimento E7-vaciar-ajuste --semilla 1",
  "--config config/instances/VN-dia-vaciar-r500.yaml --experimento E7-vaciar-ajuste --semilla 2",
  "--config config/instances/VN-dia-vaciar-r500.yaml --experimento E7-vaciar-ajuste --semilla 3"
)
New-Item -ItemType Directory -Force logs | Out-Null
$enCurso = @(); $i = 0
foreach ($args_corrida in $corridas) {
  while (($enCurso | Where-Object { -not $_.HasExited }).Count -ge 8) { Start-Sleep -Seconds 30 }
  $i++
  $enCurso += Start-Process python -ArgumentList "scripts/run_alns.py $args_corrida" -NoNewWindow -PassThru `
    -RedirectStandardOutput "logs\corrida$i.txt" -RedirectStandardError "logs\corrida${i}_err.txt"
}
$enCurso | Wait-Process
Select-String -Path logs\corrida*.txt -Pattern "Violaciones de factibilidad"
```

Cada corrida debe terminar con `Violaciones de factibilidad: 0`. Cualquier otro número es un error: avisar.

**Ojo:** las corridas escriben en las mismas carpetas que las originales y las reemplazan. Para repetirlas sin pisar los
resultados guardados, agregar `--out-dir <otra carpeta>` a cada línea.

## 5. Resultados

Cada corrida escribe en `results/<instancia>/<experimento>/`:

| Archivo | Contenido |
|---|---|
| `<variante>_s<S>.json` | resumen: buses, costo desglosado, violaciones, tiempo, commit |
| `<variante>_s<S>_buses.csv` | una fila por bus |
| `<variante>_s<S>_historia.csv.gz` | una fila por iteración del ALNS |

La variante es `vaciar-r500` en `VN-dia` y `base` en `EC-dia` y `VN-EC-dia`. Para actualizar la tabla de corridas de `results/<instancia>/README.md`:

```powershell
python scripts/comparar.py --instancia VN-dia
python scripts/comparar.py --instancia EC-dia
python scripts/comparar.py --instancia VN-EC-dia
```

