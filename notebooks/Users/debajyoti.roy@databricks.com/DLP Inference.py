# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Cluster:
# MAGIC * DBR 4.1 beta (Scala 2.11)
# MAGIC * python 3
# MAGIC * tensorflow==1.6.0
# MAGIC * Keras==2.1.5
# MAGIC * tqdm==4.20.0
# MAGIC * Pillow==5.1.0
# MAGIC * Spark Deep Learning (`PR 112`)

# COMMAND ----------

# MAGIC %md
# MAGIC # Use `sparkdl` for following tasks:
# MAGIC * Load images
# MAGIC * Load a pre-trained model checkpoint (SSD)
# MAGIC * Predict bounding boxes, classes, scores
# MAGIC 
# MAGIC Finally, use `pyspark` for insights e.g. top classes by count, which images contain a given class, and verify the results for a given image

# COMMAND ----------

import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sparkdl.image import imageIO
from sparkdl.graph.input import TFInputGraph
from sparkdl.transformers.tf_image import TFImageTransformer
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC # Images used for inference

# COMMAND ----------

from pyspark.ml.image import ImageSchema

images = ImageSchema.readImages("/mnt/roy/xview/train_chips")

images.cache()
images.count()

# COMMAND ----------

display(images)

# COMMAND ----------

# MAGIC %md
# MAGIC # Use `PIL` to decode

# COMMAND ----------

df = imageIO.readImagesWithCustomFn("/mnt/roy/xview/train_chips", imageIO.PIL_decode).cache()

# COMMAND ----------

display(
  df.select(df.image.origin, df.image.height, df.image.width, df.image.nChannels, df.image.mode)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Pre-trained model checkpoint

# COMMAND ----------

checkpoint = "/dbfs/mnt/roy/xview_model/public_release/vanilla.pb"

# COMMAND ----------

detection_graph = tf.Graph()
with detection_graph.as_default():
  image_tensor_float = tf.placeholder(tf.float32, [1, None, None, 3], name="image_tensor_float")
  image_tensor = tf.cast(image_tensor_float, tf.uint8, name="image_tensor")
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(checkpoint, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='', input_map={'image_tensor': image_tensor})

# COMMAND ----------

tfit = TFImageTransformer(
  inputCol='image',
  graph=detection_graph,
  inputTensor='image_tensor_float:0',
  outputMode='sql',
  outputMapping={
      'detection_boxes:0':'detection_boxes',
      'detection_scores:0': 'detection_scores',
      'detection_classes:0':'detection_classes'},
  channelOrder='RGB')

tfit.getOutputMapping()

# COMMAND ----------

# MAGIC %md
# MAGIC # Predict
# MAGIC * Bounding Boxes
# MAGIC * Classes
# MAGIC * Scores

# COMMAND ----------

predicted_df = tfit.transform(df)
predicted_df.cache()

# COMMAND ----------

predicted_classes_df = predicted_df.select(predicted_df.image.origin, explode(predicted_df.detection_classes).alias("class_id"))

display(
  predicted_classes_df
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Class labels

# COMMAND ----------

classes_df = spark.read.csv("/mnt/roy/xview_classes/xview_class_labels.txt", sep=":", inferSchema=True).\
  withColumnRenamed("_c0", "id").\
  withColumnRenamed("_c1", "name")
  
display(classes_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Top 10 `class`es by `count`

# COMMAND ----------

display(
  predicted_classes_df.join(classes_df, predicted_classes_df.class_id==classes_df.id).
    groupBy("name").
    count().
    orderBy(desc("count")).
    filter("count>0").
    limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # `Image`s for a given `class`

# COMMAND ----------

display(
  predicted_classes_df.
    join(classes_df, predicted_classes_df.class_id==classes_df.id).
    where("name='Passenger/Cargo Plane'")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Predictions for a given `chip`

# COMMAND ----------

fig = plt.figure()
c_1460_32 = Image.open("/dbfs/mnt/roy/xview/train_chips/1460_32.png")
plt.imshow(c_1460_32)
display(fig)

# COMMAND ----------

prediction_a_img = predicted_df.\
  where("image.origin = 'dbfs:/mnt/roy/xview/train_chips/1460_32.png'").\
  select(predicted_df.image.height, predicted_df.image.width, predicted_df.detection_classes, predicted_df.detection_scores, predicted_df.detection_boxes).\
  collect()[0]

# COMMAND ----------

clss = np.array(prediction_a_img['detection_classes'])
num_pred = len(clss)
score = np.array(prediction_a_img['detection_scores'])
box = np.reshape(prediction_a_img['detection_boxes'], (num_pred,4))

# COMMAND ----------

# MAGIC %md
# MAGIC # Draw 3 predicted boxes

# COMMAND ----------

box[:3], score[:3], clss[:3]

# COMMAND ----------

def draw_bboxes(img,boxes,classes):
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2,h2 = (img.shape[0],img.shape[1])

    idx = 0

    for i in range(len(boxes)):
        xmin,ymin,xmax,ymax = boxes[i]
        c = classes[i]

        draw.text((xmin+15,ymin+15), str(c))

        for j in range(4):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
    return source

# COMMAND ----------

arr = np.array(c_1460_32)
arr.shape

# COMMAND ----------

width,height,_ = arr.shape
cwn,chn = (300,300)
wn,hn = (int(width/cwn),int(height/chn))

# COMMAND ----------

bfull = box.reshape((wn,hn,num_preds,4))
b2 = np.zeros(bfull.shape)
b2[:,:,:,0] = bfull[:,:,:,1]
b2[:,:,:,1] = bfull[:,:,:,0]
b2[:,:,:,2] = bfull[:,:,:,3]
b2[:,:,:,3] = bfull[:,:,:,2]

bfull = b2
bfull[:,:,:,0] *= cwn
bfull[:,:,:,2] *= cwn
bfull[:,:,:,1] *= chn
bfull[:,:,:,3] *= chn
for i in range(wn):
    for j in range(hn):
        bfull[i,j,:,0] += j*cwn
        bfull[i,j,:,2] += j*cwn

        bfull[i,j,:,1] += i*chn
        bfull[i,j,:,3] += i*chn

bfull = bfull.reshape((hn*wn,num_preds,4))[0]

# COMMAND ----------

bs = bfull[score > .5]
cs = clss[score>.5]
out = draw_bboxes(arr,bs,cs)
fig = plt.figure()
plt.imshow(out)
display(fig)

# COMMAND ----------

