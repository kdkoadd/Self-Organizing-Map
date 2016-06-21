# Self-Organizing-Map
## Description:
Python/PCUDA SOM implementation

## Sample run of the "Hello World!" example of SOMs:
c:\workspace\SOM\main>python sommap.py  
Enter the desired integer associated with the runtime option below:  
1 ---Train SOM on Red, Green, Blue, Black and White RGB color data  
2 ---Train SOM on csv-formated feature data (Note this file must be stored as: ./train.csv)  
1  
Loading the training data...  
Initializing data and SOM matrix...  
-- Normalizing the data...  
-- Building the 256 by 256 SOM...  
Initialization complete  
Training the (256,256)-dimensional SOM for 10000.000000 iterations.  
-- Moving data from host to device...  
-- Starting SOM training...  
Elapsed time to train the SOM for 10000 iterations: 18.303811 secs  
### Graphical Output of sample run
![Initial Randomized Map](/InitialRandomizedSOM.png)
![Final Map](/FinalSOM.png)

