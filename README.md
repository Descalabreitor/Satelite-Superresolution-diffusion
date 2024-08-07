## Superresolución a imagenes satelitales mediante metodos de difusión
En el siguiente proyecto se exploran distintas formas de mejorar la calidad de imagenes satelitales usando modelos de difusión. 
Este proyecto corresponde con el trabajo de fin de grado con titulo "Superresolución con método generativo de difusión" de la ulpgc, eii. Por Adrián Perera Moreno

### Enviroment
Para poder usar el codigo disponible se deben instalar los paquetes dispuestos en requirements.txt mediante el siguiente comando:
```
conda create --name <env> --file requirements.txt
```
Será necesario tener disponible cuda y cudatoolkits, con las versiones correspondientes al equipo disponible.

### Datos de experimentos
Los datos de experimentos se encuentran en la carpeta "final_results" en ella se encuentran las imagenes de muestra producidas durante el entrenamiento así como los modelos guardados.

### Superresolución a imagen
Para probar los modelos se dispone del notebook "visualization_experiments" desde el cual se pueden probar comodamente todos los modelos para generar imagenes.

## Entrenamiento
Para probar modelos propios o con otros datos. Los scripts en tasks son facilmente modificables para cumplir cualquier función que se proponga. 

### Dataset

El dataset usado esta disponible en [google drive](https://drive.google.com/file/d/1M5EKvODW2XgknEIxPVwziHfFaDKWtk1U/view?usp=sharing). Para usarlo simplemente indicar su dirección en disco donde se indique.

### Implementaciones
Implementaciones:
- [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/master)
- [SRDiff](https://github.com/LeiaLi/SRDiff)
