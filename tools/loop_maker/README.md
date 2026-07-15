# loop_maker

Encuentra automaticamente una seccion de una cancion (tipicamente el coro)
que se repite musicalmente, y la exporta como un loop de audio que no
excede una duracion maxima (por defecto 30s) y sin corte audible.

## Requisitos

- `ffmpeg` instalado y en el PATH (`apt-get install -y --no-install-recommends ffmpeg`).
- `pip install -r requirements.txt`

## Uso basico

```bash
python3 make_loop.py cancion.mp3
```

Esto genera, junto al archivo de entrada (o en `--outdir`):

- `cancion_loop.mp3` — el loop final (entre `--min-duration` y `--max-duration`,
  <= 30s por defecto).
- `cancion_demo_x4.mp3` — el mismo loop repetido 4 veces seguidas, para
  confirmar de oido que la transicion no se nota.

## Opciones utiles

- `--max-duration 30` duracion maxima "objetivo" del loop en segundos.
- `--min-duration 21` duracion minima aceptable (no conviene que salga muy
  corto, tipo 12-20s).
- `--overflow 5` cuantos segundos extra se permite cruzar `--max-duration`
  (o sea, hasta 35s con los defaults) pero solo si el punto de repeticion
  natural de la cancion realmente cae ahi — no se usa "porque si".
- `--search-start` / `--search-end` acotar en que rango de la cancion
  buscar (en segundos). Por defecto busca entre el 10% y el 95% de la
  duracion total. Util si ya sabes mas o menos donde esta el coro (por
  ejemplo, despues de confirmar con un candidato de preview).
- `--energy-weight 0.4` que tanto se prefieren las secciones mas fuertes
  (los coros suelen sonar mas llenos/energicos que las estrofas).
- `--length-weight 0.35` que tanto se prefiere acercarse a `--max-duration`
  en vez de quedarse con una repeticion mas corta aunque sea un poco mas
  "perfecta". Es solo un desempate suave: nunca elige una repeticion
  claramente peor solo por ser mas larga.
- `--xfade 0.10` duracion del crossfade (segundos) en el punto de union.
- `--demo-repeats 4` cuantas veces se repite el loop en el archivo demo.
- `--bitrate 256k` bitrate del mp3 de salida.

## Como funciona (resumen del metodo)

1. Se decodifica el audio de entrada a WAV sin perdidas con `ffmpeg`, para
   poder cortar en muestras exactas despues.
2. Se analiza con `librosa`: croma (contenido armonico/melodico), beat
   tracking + onset strength (para alinear cortes al pulso), y RMS
   (energia, para preferir secciones que suenan como coro).
3. Se busca el par de instantes `(t0, t0+lag)`, con
   `min_duration <= lag <= max_duration`, cuyo contenido armonico es mas
   parecido entre si. Esto encuentra un fragmento que la propia cancion
   repite mas adelante — la mejor evidencia de que ese trozo es "loopeable"
   de forma natural, y por eso ademas suele coincidir con el coro.
4. Se ajustan `t0` y `t1` al pulso/onset musical mas cercano, para que el
   corte caiga justo en un golpe ritmico y no a mitad de una nota o palabra.
5. Se extrae el segmento exacto del WAV original.
6. Se aplica un crossfade "de loop": se toma un poco de audio real que
   continua justo despues de `t1` (la continuacion natural de la cancion,
   sin cortes) y se mezcla con el inicio del segmento. Asi, el ultimo
   sample del loop fluye hacia el primer sample igual que fluiria en la
   grabacion original, eliminando el click/salto de amplitud del corte.
   El script imprime el "salto en la union" antes y despues del crossfade
   como referencia (mientras mas chico, mejor).
7. Se exporta a mp3: el loop y una version demo repetida varias veces
   seguidas para verificar de oido.

## Limitacion importante: no "lee" la letra

El script NO hace reconocimiento de voz — no sabe literalmente que trozo es
"el coro" en terminos de letra. Lo que hace es encontrar la seccion que se
repite armonicamente y que suena mas fuerte/llena, lo cual coincide con el
coro casi siempre en musica pop, pero no esta garantizado al 100%
(a veces puede agarrar un puente o una segunda estrofa muy similar).

Se intento usar reconocimiento de voz (Whisper) para verificar la letra,
pero el modelo se descarga desde un host externo que la politica de red de
este entorno bloquea (no se debe intentar evadir ese bloqueo). Como
alternativa se probo `pocketsphinx` (no requiere descargar nada externo,
el modelo viene empaquetado por pip), pero esta pensado para voz hablada
limpia y da resultados basura sobre canto con musica de fondo — no sirve
para esto.

**Flujo recomendado cuando no estas seguro de que el resultado sea el
coro real:** generar 2-3 clips de preview (de al menos ~15-20s, ideal
21s+) en las secciones candidatas con mas repeticion/energia, y que el
usuario confirme por oido cual es el coro antes de generar el loop final
con `--search-start`/`--search-end` acotado a esa zona. Ejemplo:

```bash
ffmpeg -y -i cancion.mp3 -ss 50 -t 21 -c:a libmp3lame -b:a 192k candidato_A.mp3
```

Una vez confirmado el candidato correcto, correr `make_loop.py` con
`--search-start`/`--search-end` acotados a esa zona (por ejemplo
`--search-start 35 --search-end 85`) para que el crossfade y el ajuste
fino se calculen ahi.

## Notas para futuras canciones

- Si el resultado no cae exactamente en el coro que esperabas, prueba
  acotando `--search-start`/`--search-end` a la zona aproximada (por
  ejemplo, si sabes que el coro esta como al minuto 1:30-2:00, usa
  `--search-start 80 --search-end 140`).
- Subir `--length-weight` empuja el resultado a usar mas segundos del
  presupuesto disponible; bajarlo prioriza la coincidencia mas "perfecta"
  aunque el loop salga mas corto. A veces la repeticion natural mas parecida
  es mas corta que el objetivo (por ejemplo 25-26s en vez de 30s) — en ese
  caso no conviene forzarla a mas, porque el resto ya no encaja tan limpio.
- Si el genero tiene mucho rubato o cambios de tempo, el beat tracking
  puede fallar; en ese caso el snap a onset sigue funcionando razonablemente
  porque no depende de una grilla de tempo fija.
