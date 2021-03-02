## jsondecoder 功能:
主要是把一個影片切割成很多data(其實也不一定會成很多，總之要看那影片的總偵數和參數frame設多少)，  
又分成兩種切割方式:
* 1. 同個影片所產生的不同datas不會用到同一偵(預設是這個)  
* 2. 會用到重複的偵(要賦予shift值)  
  
## 使用說明:
### data資料夾的存放路徑:
./data/actions_names/persone_names/person_names_0000000xxxxx_xxxxxxx.json  
可以處理很多影片，只要用上述的方式存放json檔們，即可
### example
  import decodingJason as de  
  dataset_name = ["down", "run", "walk", "up", "raise"]  
  object_name = de.JasonDecoder(dataset_name, frame, shift, lowbound, nodes=25)    
  dataset, labels = object_name.decoding()  
### input:  
* dataset_name: list, store the names of actions(actions_names)  
* frame:  int, how many frames to be a data.  
* dirname: the initial directory name (e.g. train_dataset, test_dataset) 
* shift:  int, (default=1, means that 第1種的切割方式), 若要用第2種的切割方式則需要賦值(*不要賦予frame的倍數，不然將沒有意義*)  
  * e.g. 假設此影片共有200偵(0~199)，呼叫此功能, frame設30，shift設50則，會切出的資料有:  
  * 0-29, 30-59, 60-89, 90-119, 120-149, 150-179,   
  * 50-79, 80-109, 110-139, 140-169, 170-199,   
  * 100-129, 130-159, 160-189,  
  * 150-179     
* lowbound: int, (default=0)  
  * e.g. 承上面例子，若又設lowbound = 1，則會切出的資料將只有:  
  * 0-29, 30-59, 60-89, 90-119, 120-149, 150-179,   
  * 50-79, 80-109, 110-139, 140-169, 170-199,   
  * 100-129, 130-159, 160-189,  
  * ~150-179~   
  * 不會有最下面那筆，因為在那次的迴圈資料生產之前，算出若起點從150開始生產，則所生產的資料筆數只會有1，沒有>lowbound，故會跳出利用此影片生產資料的迴圈   
* nodes:  int, (default=25)  
### output:  
* dataset: 4d np.arrray, shape=(幾筆data, frame, node=25, XY=2)  
* labels:  list, shape=(幾筆data), 標籤編號是照者dataset_name所放的action順序編的，如example裡面，down就是0, run就是1, .....  
