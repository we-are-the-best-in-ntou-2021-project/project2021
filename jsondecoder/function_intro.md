## jsondecoder 功能:
主要是把一個影片切割成很多測資(其實也不一定會成很多，總之要看那影片的總偵數和參數frame設多少)，  
又分成兩種切割方式:
* 一種是同個影片所產生的不同測資們不會用到同一偵(預設是這個)  
* 另外一種就是會用到重複的(要賦予shift值)  
  
## 使用說明:
### data資料夾的存放方式:
import decodingJason as de  
object_name = de.JasonDecoder(dataset_name, frame, shift, lowbound, nodes=25)  
dataset, labels = object_name.decoding()
### input:  
dataset_name: list, store the names of actions  
frame:  
shift:  
nodes:  
### outpu:
dataset:  
labels:  
