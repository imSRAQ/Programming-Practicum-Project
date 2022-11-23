# PP-Project

# Backpropagation for two layer neural network.

This project was associated with our course work in CS571 . This project is basically about the implementation of Backpropagation algorithm for two layer neural network training and I have specifically chose my data to be binary classifiable and neural network with only one hidden layer. We have not used any library for employing the algorithm.


## Acknowledgements

 - https://www.youtube.com/watch?v=CS4cs9xVecg&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0


## Screenshots

https://images.app.goo.gl/DXsmnSbzDZ33okhEA


## Psuedo Code
1. Initially we have read the csv file.

     df=pd.read_csv("C:\\Users\hp\Downloads\Sonar221.csv")

2. Defined our neural network by defining the number of neurons in each layer.
 
       network=[n,5,1]

3. randomly assigning weights and bias to the neural network.

       for i in range(len(network) - 1):
        w = np.random.rand(network[i], network[i + 1])
        b= np.random.rand(network[i+1],1)
        weights.append(w)
        bias.append(b)
4. feedforwarding the input to the nework to calculate the error.

         
        z1=np.matmul(np.transpose(weights[0]),input)+bias[0]
        a1= _sigmoid(z1)
        z2=np.matmul(np.transpose(weights[1]),a1)+bias[1]
        a2= _sigmoid(z2)
5. Backpropagating the error by deriavting the loss function w.r.t. the weights and bias.
       
       der_L_w2=np.matmul(np.transpose(error2),np.matmul(der_a2,a1.T))
       der_L_b2=np.matmul(error2,der_a2)
       der_L_w1=np.matmul(error1.T,np.matmul(a1.T,np.matmul((one-a1),input.T)))
       der_L_b1=np.matmul(error1,der_a1)
6. Updating the weights and bias with subsequent iteration.
      
       weights[1]=weights[1]+lr*(der_L_w2.T)
       weights[0]=weights[0]+lr*(der_L_w1.T)
       bias[1]=bias[1]+lr*(der_L_b2.T)
       bias[0]=bias[0]+lr*(der_L_b1.T)
7. Again calculate the error for this.
## Results
https://images.app.goo.gl/5FsWSXJg2QKryHMv5
## Dataset Used

- sonar.csv
- Pima Indian Diabetes.csv
- Australian Credit.csv
## Authors

- Ayush Dwivedi V21093
- Syed Rizwan T22113
