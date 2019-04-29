#!/bin/bash
wget https://www.dropbox.com/s/kkg521mobrpy09r/epoch-9.ckpt?dl=1
mv epoch-9.ckpt?dl=1 model/submission/ckpts/epoch-9.ckpt
wget https://www.dropbox.com/s/su88eqsl23t4ooh/elmo_model_adap_93000.tar?dl=1
mv elmo_model_adap_93000.tar?dl=1 elmo_model_adap_93000.tar