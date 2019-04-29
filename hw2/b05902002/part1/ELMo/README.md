3. # ADL HW2 ELMo Model Training

    > B05902002 資工三 李栢淵
    
    
    
    ### Training and Testing 
    
    ##### Training
    
    ```sh
    CUDA_VISIBLE_DEVICES=0 python elmo_main.py -b 32 -e 20 -p 3200 -md elmo_model_adap_93000.tar
    ```
    
    
    
    ##### Plotting Figure 
    
    After training, there will exist two npy files in the current directory.
    
    ```sh
    python plot_graph.py
    ```
