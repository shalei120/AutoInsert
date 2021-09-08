# AutoInsert
## Inference 
* Task 1:
    ```
    python3 main_mt.py -m transformer -b 64  -d DE_EN -g 0 -layer 6 -emb 512 -r False
    ```
* Task 2:
    ```
    python3 main_mt.py -m transformer -b 64  -d DE_EN -g 0 -layer 6 -emb 512 -t hard -r False
    ```

## Train

* 1, Put the dataset dataset_v1 under the `./.data` directory
* 2, Follow the shell in `./submitjade` directory
    - aitf.sh is  task 1
    - aitf_hard.sh is task 2
    
### Run in command

If you would like to directly run the training process:

* Task 1: aitf.sh

    - First, create data from raw dataset
    
        ```
        python3 textdata.py
        ```
    - Second, preprocess
        ```
        TEXT=artifacts/fsdata
        export PYTHONPATH=$PYTHONPATH:$PWD/fairseq
        python3 fairseq/fairseq_cli/preprocess.py --source-lang de --target-lang en \
            --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
            --destdir artifacts/preprocessed/ \
            --workers 20
        ```
     - Third, run training

        ```
        python3 main_mt.py -m transformer -b 64  -d DE_EN -g 0 -layer 6 -emb 512 
        ```
* Task 2: aitf_hard.sh
    - First, create data from raw dataset
        ```
        python3 textdata.py
        ```
    - Second, preprocess
        ```
        TEXT=artifacts/fsdata_hard
        export PYTHONPATH=$PYTHONPATH:$PWD/fairseq
        python3 fairseq/fairseq_cli/preprocess.py --source-lang de --target-lang en \
            --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
            --destdir artifacts/preprocessed/ \
            --workers 20
        ```
     - Third, run training

        ```
        python3 main_mt.py -m transformer -b 64  -d DE_EN -g 0 -layer 6 -emb 512 -t hard
        ```


    
    

