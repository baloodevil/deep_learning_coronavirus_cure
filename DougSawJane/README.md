# Hello World example of LSTM
Modeled after Brandon Rohrer's [excellent video](https://www.youtube.com/watch?v=WCUNPb-5EYI) on the topic.

## All possible sentences...
Doug saw Jane.  
Doug saw Spot.  
Jane saw Doug.  
Jane saw Spot.  
Spot saw Jane.  
Spot saw Doug.  

## So the dictionary will be...
Jane	→	`1 0 0 0 0`  
Doug	→	`0 1 0 0 0`  
Spot	→	`0 0 1 0 0`  
saw	    →	`0 0 0 1 0`  
.	    →	`0 0 0 0 1`  


## One hot encoding each sentence (including the period)...
Doug saw Jane.
````
0 1 0 0 0

0 0 0 1 0

1 0 0 0 0

0 0 0 0 1
````

Doug saw Spot.
````
0 1 0 0 0

0 0 0 1 0

0 0 1 0 0

0 0 0 0 1
````

Jane saw Doug.
````
1 0 0 0 0

0 0 0 1 0

0 1 0 0 0

0 0 0 0 1
````

Jane saw Spot.
````
1 0 0 0 0

0 0 0 1 0

0 0 1 0 0

0 0 0 0 1
````

Spot saw Jane.
````
0 0 1 0 0

0 0 0 1 0

1 0 0 0 0

0 0 0 0 1
````

Spot saw Doug.
````
0 0 1 0 0

0 0 0 1 0

0 1 0 0 0

0 0 0 0 1
````

Encoding each sentence with a single character (using 'P' for spot and 's' for saw.  not including spaces)...  

`DsJ.DsP.JsD.JsP.PsJ.PsD.`


Mistakes a simple RNN could make...  
`Doug saw Doug.`  
`Doug saw Jane saw Spot saw Doug saw Jane`...  
`Doug. Jane. Spot.`  

An LSTM introduces three things to address this...  
1. Forget gate
2. Memory
3. 
