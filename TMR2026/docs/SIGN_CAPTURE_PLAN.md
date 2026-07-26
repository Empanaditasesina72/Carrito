# Plan de captura para adaptar el detector a la pista real

**Para mañana en la noche.** Sigue esto tal cual y el entrenamiento de la noche
siguiente parte de datos del dominio real, que es lo único que falta.

---

## Por qué hace falta (el número que lo justifica)

Medido esta noche sobre `car_hard` — 294 fotos de pista degradadas
sintéticamente (ruido de ganancia 22, oscurecimiento gamma, desenfoque de
movimiento, pérdida de resolución):

| | `stop`/`red` R@0.55 | confianza media |
|---|---|---|
| Benchmark sintético | **96.3 %** | **0.77** |
| **Cámara real del carro** | detecta | **0.61** |

El modelo resuelve al 96 % todo lo que sé simular. En la cámara real la
confianza cae a 0.61, apenas **0.06 arriba** del umbral de 0.55 que frena el
carro. **La degradación que puedo inventar no es la que falla.** Lo que difiere
es lo que no puedo sintetizar: tu señal impresa concreta, tu superficie de pista
(plástico negro con líneas blancas — el dataset es verde/gris), y la respuesta
de color y ruido de este sensor específico.

Eso solo se cierra con fotos reales.

---

## Requisito duro: solo la cámara del Pi

> **No sirven fotos de la Sony ni de la cámara de acción.** Otro sensor tiene
> otra óptica, otro balance de blancos, otro ruido y otro campo de visión. Un
> modelo entrenado con esas fotos vuelve a tener brecha de dominio, solo que
> ahora escondida. Las fotos tienen que pasar por el mismo `CameraStream` que
> usa el carro.

Las cámaras buenas sí sirven — para el **paper**: fotos del vehículo y de la
pista para las figuras. Ese es otro uso, no entrenamiento.

---

## Antes de empezar

```bash
sudo systemctl stop carrito_tmr
```

El servicio toma la cámara; si está corriendo, `capture_track.py` falla.

Y confirma que la pista esté como va a competir: misma iluminación, misma
posición de las cajas del estacionamiento, señal a 1.5 m del inicio.

---

## Cómo capturar: pasadas rodando, no fotos fijas

Poner el carro en distancias exactas es lento y da pocas imágenes. En vez de
eso: **arranca la captura automática y empuja el carro despacio hacia la señal.**
Una pasada de 2.0 m a 0.2 m en unos 20 s da ~28 cuadros que barren todo el rango
de distancias, y de paso incluyen el desenfoque de movimiento real.

```bash
cd ~/Carrito
python TMR2026/tools/capture_track.py --auto 0.7
```

Corta con **Ctrl+C** al terminar cada pasada (nunca Ctrl+Z — deja la cámara
tomada). Las imágenes se acumulan en `TMR2026/tools/captures/`.

### Lista de pasadas

Con la **señal de ALTO**, que es la que frena el carro:

| # | Pasadas | Variación |
|---|---|---|
| 1–3 | 3 | Centrado, de frente. Recto hacia la señal |
| 4–5 | 2 | Carro desplazado ~10 cm a la **izquierda** del centro |
| 6–7 | 2 | Carro desplazado ~10 cm a la **derecha** |
| 8–9 | 2 | Señal **girada ~20°**, no de frente a la cámara |
| 10–11 | 2 | **Luz distinta** a la de las demás (más tenue, o una lámpara movida) |

Con las **otras señales** (verde, rojo, amarillo, izquierda, derecha, recto),
si las tienes físicamente:

| # | Pasadas | Variación |
|---|---|---|
| 12–17 | 1 por señal | Centrada, de frente |

Y **negativos** — igual de importantes, para que no alucine:

| # | Pasadas | Variación |
|---|---|---|
| 18–20 | 3 | Pista **sin ninguna señal**. Incluye las cajas del estacionamiento, muebles, lo que se vea |

**Total: ~20 pasadas, unos 15–20 minutos, ~500 cuadros.** Eso alcanza de sobra:
el dataset actual son 1029 imágenes, así que 500 reales pesan lo suficiente para
mover el modelo hacia tu dominio sin borrar lo que ya sabe.

---

## Ponle nombre a cada grupo

Para poder etiquetar por lotes, mueve los cuadros a carpetas por señal en
cuanto termines cada grupo:

```bash
cd ~/Carrito/TMR2026/tools/captures
mkdir -p stop green red yellow left right straight nada
```

…y mueve ahí lo de cada pasada. Si se te olvida, no es fatal — se puede separar
después, solo es más lento.

---

## Súbelo a la PC

```bash
cd ~/Carrito && git add -A TMR2026/tools/captures && git commit -m "Add real track captures for detector fine-tuning" && git push origin main
```

Si el push a GitHub va lento, pásalas directo por LAN:

```bash
scp -r ~/Carrito/TMR2026/tools/captures angel01@PC:/tmp/
```

Avísame cuando estén y yo hago el resto.

---

## Qué hago yo con ellas

1. **Auto-etiquetado** con el modelo actual como maestro
   (`tools/autolabel_signs.py --conf 0.10`), umbral bajo a propósito para que
   proponga hasta las detecciones dudosas.
2. **Revisión manual** — genero una hoja de contactos con las cajas dibujadas y
   corrijo lo que salga mal. Este paso no se salta: etiquetas malas entrenan un
   modelo malo, y el auto-etiquetado hereda los errores del maestro.
3. **Mezcla con peso**: las capturas reales repetidas para que pesen parecido al
   dataset sintético, sin que este las ahogue.
4. **Fine-tune** sobre el ganador de esta noche, y **medición en `car_hard` más
   un benchmark nuevo hecho con tus propias fotos** — ese sí mide lo que
   importa.

---

## Lo que NO hay que hacer

- ❌ Fotos con celular, Sony o cámara de acción para entrenar.
- ❌ Capturar con el servicio `carrito_tmr` corriendo.
- ❌ Ctrl+Z para cortar (deja el GPIO y la cámara tomados; luego sale
  `lgpio.error: 'GPIO not allocated'`).
- ❌ Cambiar la iluminación a media pasada sin anotarlo — si no sé qué
  condición es cada grupo, no puedo balancear el conjunto.
- ❌ Borrar los negativos por parecer "fotos vacías". Son los que evitan los
  frenazos fantasma.
