# Costo de la partición y cotas de optimalidad

El proyecto no resuelve los 64 800 viajes del día de una vez: asigna cada recorrido al electroterminal más cercano y
resuelve cada terminal por separado. Esa partición prohíbe que un bus de un terminal haga viajes de otro, así que el
óptimo del problema completo es siempre **menor o igual** que la suma de los óptimos por terminal. La diferencia es el
**costo de la partición**.

Este documento reúne:

1. el experimento exacto que lo mide en instancias chicas (`particion-chica/E1-x35-s1`, unas 8 horas de cómputo);
2. las cotas inferiores de flota y de costo definidas para el proyecto, y sus valores en cada instancia;
3. qué se concluye de ambos;
4. una opinión sobre la optimalidad de los resultados y sobre qué hacer después.

Todos los costos están en USD. Los números se recalcularon desde los JSON de `results/` (que no se versionan) el
2026-10-16.

---

## 1. El experimento: `particion-chica/E1-x35-s1`

### Diseño

- **Pregunta:** ¿cuánto se pierde por resolver cada electroterminal por separado?
- **Muestra:** 35 viajes al azar de cada uno de los 5 electroterminales (semilla 1), tomados del día hábil completo.
- **Instancias:** los 5 terminales solos (VN, EC, LE, LR, SR), los pares VN-EC y LE-SR, y los 5 juntos (175 viajes, 5
  depósitos). Cada instancia de varios terminales es exactamente la unión de las muestras de sus partes, así que las
  comparaciones son directas.
- **Método:** el MILP exacto (Gurobi) en las 8 instancias, con 1 hora por instancia. Si no llega al óptimo, se guardan la
  mejor solución y la cota inferior que entrega Gurobi.
- **Tiempo total:** 8,03 h (8 MILP que agotaron su hora).
- **Config y script:** `config/instances/particion-chica.yaml`, `scripts/costo_particion.py`.

### Resultados por instancia

| Instancia | Viajes | Depósitos | Variables / restricciones | Mejor solución: buses | Mejor solución: costo | Cota de Gurobi | Gap | Cota de flota | Cota de costo |
|---|---|---|---|---|---|---|---|---|---|
| VN | 35 | 1 | 11 597 / 20 812 | 8 | 2 277,5 | 2 227,5 | 2,2 % | 8 | 2 112,9 |
| EC | 35 | 1 | 13 145 / 23 705 | 8 | 2 332,8 | 2 294,5 | 1,6 % | 8 | 2 105,7 |
| LE | 35 | 1 | 12 457 / 22 591 | 8 | 2 323,4 | 2 301,1 | 1,0 % | 8 | 2 111,2 |
| LR | 35 | 1 | 10 991 / 19 842 | 8 | 2 278,5 | 2 242,6 | 1,6 % | 8 | 2 088,6 |
| SR | 35 | 1 | 14 326 / 25 539 | 7 | 2 007,0 | 1 966,1 | 2,0 % | 7 | 1 828,2 |
| VN-EC | 70 | 2 | 61 046 / 105 578 | 12 | 3 563,6 | 3 367,5 | 5,5 % | 12 | 3 218,6 |
| LE-SR | 70 | 2 | 65 006 / 112 126 | 13 | 3 778,8 | 3 618,4 | 4,2 % | 13 | 3 439,5 |
| Los 5 | 175 | 5 | 595 582 / 1 000 219 | 28 | 8 204,4 | 7 651,8 | 6,7 % | 28 | 7 496,7 |

Ninguna solución tiene violaciones de factibilidad. Desglose del costo de las mejores soluciones:

| Instancia | Costo fijo de buses | Deadhead | Carga diurna | Carga nocturna | Cargas de día | Viajes por bus |
|---|---|---|---|---|---|---|
| VN | 2 000 | 134,4 | 29,9 | 113,1 | 9 | 4,4 |
| EC | 2 000 | 185,6 | 17,4 | 129,8 | 6 | 4,4 |
| LE | 2 000 | 173,4 | 5,1 | 145,0 | 2 | 4,4 |
| LR | 2 000 | 155,2 | 2,7 | 120,7 | 1 | 4,4 |
| SR | 1 750 | 146,1 | 0,0 | 111,0 | 0 | 5,0 |
| VN-EC | 3 000 | 281,9 | 49,8 | 231,9 | 13 | 5,8 |
| LE-SR | 3 250 | 277,2 | 0,0 | 251,6 | 0 | 5,4 |
| Los 5 | 7 000 | 578,2 | 13,5 | 612,7 | 3 | 6,3 |

### Pérdida por partir

Como ningún MILP cerró, la pérdida en costo queda en un intervalo:

- **mínimo** = suma de las cotas de Gurobi de las partes − mejor solución del conjunto;
- **máximo** = suma de las mejores soluciones de las partes − cota de Gurobi del conjunto.

| Conjunto | Buses por separado | Buses juntos | **Flota mínima ahorrada** | Pérdida en costo | En % del costo junto | Pérdida sin contar los buses | Deadhead separado → junto |
|---|---|---|---|---|---|---|---|
| VN + EC | 8 + 8 = 16 | 12 | **4 (25 %)** | 958 – 1 243 | 27 – 35 % | −42 – 243 | 320 → 282 (−12 %) |
| LE + SR | 8 + 7 = 15 | 13 | **2 (13 %)** | 488 – 712 | 13 – 19 % | −12 – 212 | 319 → 277 (−13 %) |
| Los 5 | 39 | 28 | **11 (28 %)** | 2 827 – 3 568 | 34 – 43 % | 77 – 818 | 795 → 578 (−27 %) |

"Pérdida sin contar los buses" es el intervalo de pérdida menos 250 USD por cada bus de la flota mínima ahorrada.

---

## 2. Las cotas de optimalidad

Una **cota inferior** `L` es un valor que ninguna solución factible puede mejorar. Si una solución cuesta `C`, le falta
**a lo más** `(C − L) / L` para el óptimo. Sirven porque en los días completos no se puede obtener el óptimo: el MILP de
`VN-dia` tendría unos 22 millones de pares de viajes encadenables. Código: `src/ebus/evaluation/metrics.py`
(`peak_concurrent_trips`, `min_buses_con_deadheads`, `cota_costo`) y `scripts/cota.py`.

### Cota de flota 1: viajes simultáneos

El máximo número de viajes en curso en un mismo minuto. Un bus hace un viaje a la vez, así que ninguna solución usa menos
buses. Es instantánea de calcular, pero ignora que entre dos viajes hay que moverse.

### Cota de flota 2: con deadheads (la que se usa)

El mínimo número de buses que cubren todos los viajes si un bus puede hacer `b` después de `a` cuando
`salida(b) − llegada(a) ≥ deadhead mínimo de fin(a) a inicio(b)` (directo o pasando por un depósito, el menor),
**ignorando la batería, los cargadores y el horizonte**.

- **Por qué es cota:** las cadenas de cualquier solución factible cubren los viajes con caminos de ese grafo de
  encadenamientos. Como el grafo admite más encadenamientos que los factibles de verdad, su mínimo de caminos es menor o
  igual que la flota de cualquier solución.
- **Cálculo:** mínimo de caminos = viajes − emparejamiento máximo. Se resuelve como un flujo máximo sobre líneas de tiempo
  por estación, sin listar los pares. Tarda de 47 s (`LR-dia`) a 38 min (`VN-EC-dia`).

### Cota de costo

> `cota = B_min × 250 USD + E_min × 0,10 USD/kWh`

| Término | Valor | Por qué ninguna solución puede tener menos |
|---|---|---|
| `B_min` | la cota de flota con deadheads | ninguna solución usa menos buses |
| 250 USD | costo fijo por bus y día | es el mismo para todo bus usado |
| `E_min` | energía de todos los viajes (km × 1,4 kWh/km), más lo que falte para llenar la batería si los buses parten bajo 100 % | lo gastado se repone al final del día, y la energía de los viajes es la misma en toda solución |
| 0,10 USD/kWh | precio mínimo de la tarifa | toda la energía se paga a un precio mayor o igual |

Ignora el deadhead (su costo y su energía), la batería, los cargadores y que la carga de día es más cara.

### Valores por instancia

Brecha = (mejor solución − cota) / cota. "Mejor solución" es la más barata guardada en `results/`.

| Instancia | Viajes | Depósitos | Viajes simultáneos | **Cota de flota** | Energía de los viajes (kWh) | **Cota de costo** | Mejor solución | Brecha en buses | Brecha en costo |
|---|---|---|---|---|---|---|---|---|---|
| VN-10 (bloque) | 536 | 1 | 411 | **422** | 44 722¹ | **109 972** | 422 buses, 114 365 (ALNS) | 0 | 4,0 % |
| `VN-dia` | 7 448 | 1 | 769 | **774** | 206 799 | **214 180** | 783 buses, 228 313 (ALNS) | +9 (1,2 %) | 6,6 % |
| `EC-dia` | 17 288 | 1 | 2 063 | **2 079** | 595 939 | **579 344** | 2 168 buses, 651 251 (ALNS) | +89 (4,3 %) | 12,4 % |
| `LR-dia` | 8 496 | 1 | 801 | **805** | 208 091 | **222 059** | 828 buses, 240 712 (baseline) | +23 (2,9 %) | 8,4 % |
| `LE-dia` | 13 590 | 1 | 1 483 | **1 496** | 446 336 | **418 634** | sin corrida | | |
| `SR-dia` | 17 978 | 1 | 1 610 | **1 623** | 432 900 | **449 040** | sin corrida | | |
| `VN-EC-dia` | 24 736 | 2 | 2 809 | **2 833** | 802 738 | **788 524**² | 2 939 buses, 874 609 (ALNS) | +106 (3,7 %) | 10,9 % |
| `LE-SR-dia` | 31 568 | 2 | 3 073 | **3 096** | 879 236 | **861 924**² | sin corrida | | |
| `global-dia` | 64 800 | 5 | 6 609 | pendiente | | | sin corrida | | |

¹ Incluye lo que falta para llenar la batería de 422 buses que parten con 80 %.
² La cota de flota de las instancias de dos terminales se calculó aparte; no tienen `cota.json`.

---

## 3. Análisis de los resultados

### Lo que muestra el experimento chico

- **El MILP no cierra ni con 35 viajes y un depósito en 1 hora** (gap 1,0–2,2 %). Con dos depósitos sube a 4–5,5 %, y con
  cinco a 6,7 % sobre 600 mil variables y 1 millón de restricciones. Cada depósito agrega su red de carga de todo el día.
- **Las 8 soluciones alcanzan exactamente la cota de flota con deadheads.** Como ninguna solución puede usar menos buses,
  la flota mínima de cada instancia queda demostrada aunque el MILP no haya cerrado el costo. En estas instancias la batería
  no impidió llegar a la cota.
- **La diferencia de cotas predice exactamente la flota ahorrada.** En los tres conjuntos, la suma de las cotas de flota por
  separado menos la del conjunto da 4, 2 y 11 buses: lo mismo que el MILP. La diferencia de cotas de costo da 1 000, 500 y
  2 750 USD, que es exactamente esa flota por 250, porque la energía de los viajes no cambia al juntar.
- **Casi toda la pérdida demostrable es flota.** Descontando los buses ahorrados, la pérdida en VN-EC y LE-SR va de un poco
  negativa a ~240 USD, así que ni siquiera se puede asegurar que juntar baje los demás costos. Con los 5 juntos sí: al menos
  77 USD más, con el deadhead bajando 27 %.
- **Las magnitudes no se extrapolan.** La muestra es rala: 35 viajes repartidos en todo el día dan 4–6 viajes por bus,
  contra 8–9,5 en los días completos (`VN-dia` 9,5; `VN-EC-dia` 8,4; `EC-dia` 8,0). Con tantos huecos, un bus de otro terminal tiene mucho que llenar.

### Lo que muestran las cotas en los días completos

- **En buses, las soluciones están a 1–4 % de la cota,** y en VN-10 la alcanzan (422 = 422). La cota de flota es ajustada.
- **En costo, la brecha es de 4–12 %, pero buena parte no es recuperable.** La cota no cobra el deadhead, y ninguna solución
  real lo tiene en cero: en `EC-dia` el deadhead de la mejor solución son ~38 000 USD, más de la mitad de la brecha. En las
  instancias chicas, la cota de Gurobi (que es menor o igual que el óptimo) ya queda 5–9 % sobre la cota de costo, así que
  esa es aproximadamente la parte de la brecha que la cota no ve.
- **`EC-dia` y `VN-EC-dia` son donde hay más margen:** 89 y 106 buses sobre la cota, y en sus corridas la flota seguía
  bajando al final.

### El costo de la partición a escala real (par VN-EC)

| Medida | Qué es | Resultado |
|---|---|---|
| (a) Pérdida total acotada | mejores soluciones por separado (228 313 + 651 251 = 879 564) contra la cota de `VN-EC-dia` (788 524) | partición + heurística ≤ **11,5 %** (≤ 118 buses) |
| (b) Diferencia de cotas | 774 + 2 079 = 2 853 buses contra 2 833 juntos | **20 buses; 5 000 USD (0,63 %)** |
| (c) ALNS separado contra junto | promedio de 3 semillas | **~10 buses; 4 520 USD (0,52 %)** |

La medida (a) es rigurosa pero floja, porque la cota ignora el deadhead. La (b) resultó exacta en todas las instancias
chicas. La (c) es lo que efectivamente se observa con la heurística.

---

## 4. Opinión y análisis de optimalidad

Esta sección es mi lectura de los resultados (Claude, 2026-10-16), no un resultado del equipo.

**1. Partir por electroterminal es una buena decisión a escala real.** En el único par medido a escala, VN-EC, todas las
medidas coinciden en una pérdida del orden de 0,5–0,7 % del costo (10–20 buses de casi 3 000). Es menos que la brecha de
la heurística respecto de la cota de flota en `EC-dia` (4,3 % en buses). El costo de partir es menor que lo que todavía
se puede ganar mejorando el ALNS en las instancias grandes; ahí conviene poner el esfuerzo.

**2. El experimento chico demuestra bien el mecanismo y la calidad de la cota de flota, pero no la magnitud.** Sus 25–28 %
de flota perdida son un efecto de la muestra rala, no del problema real. Por diseño, 35 viajes al azar de todo el día
dejan buses casi vacíos. Si se quiere un número exacto más representativo, propongo muestrear **ventanas densas**: todos
los viajes de unas pocas líneas de cada terminal en un bloque de 1–2 horas, que conserva la densidad real (viajes por bus y
huecos entre viajes). Además, todo descansa en **una sola semilla**: con 2 o 3 semillas más se sabría si los 4, 2 y 11
buses son estables.

**3. Una precisión sobre la "flota exacta".** Lo demostrado es que la **flota mínima** baja 4, 2 y 11 buses. Que el
**óptimo de costo** use esa flota mínima es otra afirmación:
- En las 5 instancias de un terminal se puede demostrar. Una solución con un bus más costaría al menos la cota de costo
  más 250, y eso supera la mejor solución encontrada (por ejemplo, en VN: 2 112,9 + 250 = 2 362,9 > 2 277,5). Así que su
  óptimo usa la flota mínima.
- En las 3 instancias conjuntas, no: en VN-EC, 3 218,6 + 250 = 3 468,6 < 3 563,6. En principio, el óptimo de costo junto
  podría usar 13 buses.

Es improbable: el bus extra tendría que ahorrar más de 250 USD en deadhead y carga, cuando todo el deadhead de la mejor
solución de VN-EC son 282 USD. Pero no está demostrado, así que en el informe conviene decir "flota mínima ahorrada" y no
"flota del óptimo ahorrada".

**4. La cota de flota es la herramienta más valiosa del proyecto; la cota de costo es la más débil.** La de flota fue exacta
en las 8 instancias chicas y en VN-10, y queda a 1–4 % en los días completos. La de costo queda 8–11 % bajo las mejores
soluciones en las instancias chicas, casi todo por no cobrar el deadhead. Se puede fortalecer sin perder validez:
convertir el mismo flujo de la cota de flota en un **flujo de costo mínimo**. Habría que cobrar 250 por cada bus que sale y
el costo (y la energía) del deadhead en cada encadenamiento, la salida y el retorno, ignorando igual la batería y los
cargadores. Seguiría siendo una cota inferior y cobraría el deadhead mínimo, que es la mayor parte de la brecha actual.
Con eso, las brechas de 6,6–12,4 % de los días completos probablemente bajarían a unos pocos puntos, y se sabría cuánto le
queda de verdad al ALNS.

**5. El MILP no sirve como referencia exacta, ni en instancias chicas.** Si en 1 hora no cierra 35 viajes, la comparación
"ALNS contra óptimo" en instancias de 40–100 viajes solo dará intervalos. Para certificar optimalidad, rinden más las cotas
combinatorias (flota exacta, y el flujo de costo mínimo propuesto) que más horas de Gurobi.

**6. Lo que falta para cerrar la pregunta a escala real.** Solo el par VN-EC está medido con soluciones. Faltan LE-SR y,
sobre todo, los 5 juntos: en las instancias chicas la pérdida de los 5 (28 %) fue mayor que la de los pares (13–25 %), así
que la pérdida real de partir en 5 puede ser mayor que el 0,5–0,7 % de un par. La cota de flota de `global-dia` (unas 1–5
horas de cálculo) daría, con la medida (b), una estimación que en las instancias chicas resultó exacta.

**7. Reproducibilidad.** Las 8 corridas registran el commit `34faa8f` con `cambios_sin_commitear: true`, y ese commit
pertenece al historial del repositorio original, no a este. Las soluciones pasaron el validador y su costo coincide con el
objetivo de Gurobi, así que los números son confiables. Pero si se repite el experimento, conviene hacerlo con el árbol
limpio.

---

## 5. Cómo reproducir

```powershell
python scripts/costo_particion.py --config config/instances/particion-chica.yaml   # ~8 h: 8 MILP de hasta 1 h
python scripts/cota.py --config config/instances/VN-dia.yaml                       # cota de una instancia
```

Las salidas quedan en `results/particion-chica/E1-x35-s1/` (un JSON por instancia y `resumen.json`) y en
`results/<instancia>/cota.json`.
