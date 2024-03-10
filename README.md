# Latent Diffusion Models
This is the repository for a DTU master thesis with the title Age Manipulation with Latent Diffusion Models. This repository goes in hand with the written thesis. 

### Project structure: 
Files titled as inference are for the inference stage of the project and files titled as jobscript are the jobscript that is sent to the DTU servers for training/inference. 

Generated data is accessible with this link: https://drive.google.com/drive/folders/1B71jXGWRW36pgbQnXfkReZO7k7YtWXWj?usp=sharing 

Evaluations folder includes the calculation done in the evaluation chapter in the thesis. 

The config files used are under configs -> latent-diffusion -> All config files with agemodel in the title are specific for this project
To look for specific parts in the project that can be done with the config file. 

The data handling and conditioning is under ldm -> data -> agenet.py

The AgeEmbedder that processed the conditioning is under ldm -> modules -> encoders -> models.py

s212461
