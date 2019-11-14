#!/bin/bash
cd 1_layer_128_nodes
python ../mido_tonotes.py notes_result
cd ..  
cd 2_layer_64_nodes   
python ../mido_tonotes.py notes_result
cd ..
cd 4_layer_128_nodes
python ../mido_tonotes.py notes_result
cd ..
cd 1_layer_64_nodes   
python ../mido_tonotes.py notes_result
cd ..
cd 3_layer_128_nodes  
python ../mido_tonotes.py notes_result
cd ..
cd 4_layer_64_nodes
python ../mido_tonotes.py notes_result
cd ..
cd 2_layer_128_nodes  
python ../mido_tonotes.py notes_result
cd ..
cd 3_layer_64_nodes
cd ..
