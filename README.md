# Tensorbot_NLP_System

## Reinforcement Learning System for Chatbot
![image](./images/rl_chatbot.jpg)


## Transformer Block
![image](./images/bart.jpg)




## Traing Results
- I trained these results on the SQUAD set
## After 100 epoch
![image](./images/mlflow_train.jpg)

### Tunning model with Reinforcement Learning
![image](./images/mlflow_train_rl.jpg)

### Result
```  
> What could overgenerous outside ornaments be sometimes?
= I am impossible answer this question
< I am impossible answer this question
Bleu score  1.0
-----------------------------------------
> What provides a bridge between the different dungeons?
= overworld
< overworld
Bleu score  1.0
-----------------------------------------
> Newborns are particularly susceptible to infections caused by?
= low virulence organisms like Staphylococcus and Pseudomonas
< low virulence organisms like Staphylococcus and Pseudomonas
Bleu score  1.0
-----------------------------------------
```