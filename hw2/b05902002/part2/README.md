3. # ADL HW2 BERT Model Training

    > B05902002 資工三 李栢淵
    
    
    
    ### Training and Testing 
    
    ##### Training and Testing
    
    ```sh
    CUDA_VISIBLE_DEVICES=0 python main.py -dp /PATH/TO/CSVs/DIRECTORY/ -tp /PATH/TO/TEST_CSV -lr 5e-6 -p 2220 -b 8 -tr 0 -e 10 -md MODEL_NAME -l 64 -bert bert-large-uncased
    ```
    
    ##### Just Testing
    
    ```sh
    CUDA_VISIBLE_DEVICES=0 python main.py -tp /PATH/TO/TEST_CSV -lr 5e-6 -p 2220 -b 8 -tr 0 -e 10 -md MODEL_NAME -l 64 -o OUTPUT_CSV -bert bert-large-uncased 
    ```
