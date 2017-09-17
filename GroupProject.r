###GROUP PROJECT###
setwd("C:/Users/rasadigov/Documents/ClassProjects/BUAN6356-GR") # @work
setwd("C:/Users/rasha/Dropbox/UTD/BUAN 6356/BUAN6356-GR") # @home

library(mice)
library(rpart)
library(caret)
library(e1071)
library(randomForest)
library(pROC)
library(ada)
library(DMwR)
library(ROCR)

bank=read.csv("bank-full.csv",sep = ",")
banktest=read.csv("bank.csv", sep=";")
View(bank)
View(banktest)
#EAD of the DataSet
md.pattern(bank) #no missing data
barplot(height=cbind(sum(bank$y=="yes"),sum(bank$y=="no")), axisnames=TRUE, 
	main="Barplot", names.arg=c("Yes", "No"), col=c("red", "blue"), beside=TRUE)
table(bank$y)


###Simple Tree Model
BankTree=rpart(y~age+job+marital+education+default+balance+
		housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome, data=bank)

pred=predict(BankTree, bank, type=c("class"))
cmatrix=confusionMatrix(bank$y, pred)
cmatrix$table

roc=prediction(as.numeric(pred), bank$y)
perf=performance(roc, "tpr", "fpr")
plot(perf, col="blue", lwd=1)
abline(0,1, col="red")

auc=performance(roc, "auc")
auc

#Testing the Small Data with the Tree Model
pred1=predict(BankTree, banktest, type=c("class"))
cmatrix1=confusionMatrix(banktest$y, pred1)
cmatrix1$table

roc1=prediction(as.numeric(pred1), banktest$y)
perf1=performance(roc1, "tpr", "fpr")
plot(perf1, col="blue", lwd=1)
abline(0,1, col="red")

auc1=performance(roc1, "auc")
auc1

###Naive Bayes Classifier
BankNbmodel=naiveBayes(y~age+job+marital+education+default+balance+
		housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome, data=bank)

pred2=predict(BankNbmodel, bank, type=c("class"))
cmatrix2=confusionMatrix(bank$y, pred2)
cmatrix2$table

roc2=prediction(as.numeric(pred2), bank$y)
perf2=performance(roc2, "tpr", "fpr")
plot(perf2, col="blue", lwd=1)
abline(0,1, col="red")

auc2=performance(roc2, "auc")
auc2

pred3=predict(BankNbmodel, banktest, type=c("class"))
cmatrix3=confusionMatrix(banktest$y, pred3)
cmatrix3$table

roc3=prediction(as.numeric(pred3), banktest$y)
perf3=performance(roc3, "tpr", "fpr")
plot(perf3, col="blue", lwd=1)
abline(0,1, col="red")

auc3=performance(roc3, "auc")
auc3

###Random Forest Model
bankrf=randomForest(y~age+job+marital+education+default+balance+
	housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome, 
	data=bank, ntree=100, mtry=7,importance=TRUE, na.action=na.roughfix, replace=FALSE)

pred4=predict(bankrf, bank, type=c("class"))
cmatrix4=confusionMatrix(bank$y, pred4)
cmatrix4$table

roc4=prediction(as.numeric(pred4), bank$y)
perf4=performance(roc4, "tpr", "fpr")
plot(perf4, col="blue", lwd=1)
abline(0,1, col="red")

auc4=performance(roc4, "auc")
auc4

pred5=predict(BankNbmodel, banktest, type=c("class"))
cmatrix5=confusionMatrix(banktest$y, pred5)
cmatrix5$table

roc5=prediction(as.numeric(pred5), banktest$y)
perf5=performance(roc5, "tpr", "fpr")
plot(perf5, col="blue", lwd=1)
abline(0,1, col="red")

auc5=performance(roc5, "auc")
auc5

###Boosted Model
bankada=ada(y~age+job+marital+education+default+balance+
              housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome, data=bank,
            control=rpart.control(maxdepth=30,cp=0.010000,minsplit=20,xval=10),iter=50)

pred6=predict(bankada, bank)
cmatrix6=confusionMatrix(bank$y, pred6)
cmatrix6$table

roc6=prediction(as.numeric(pred6), bank$y)
perf6=performance(roc6, "tpr", "fpr")
plot(perf6, col="blue", lwd=1)
abline(0,1, col="red")

auc6=performance(roc6, "auc")
auc6

pred7=predict(bankada, banktest)
cmatrix7=confusionMatrix(banktest$y, pred7)
cmatrix7$table

roc7=prediction(as.numeric(pred7), banktest$y)
perf7=performance(roc7, "tpr", "fpr")
plot(perf7, col="blue", lwd=1)
abline(0,1, col="red")

auc7=performance(roc7, "auc")
auc7

#Logistic Regression
banklogit=glm(y~age+job+marital+education+default+balance+
                housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome, 
              family=binomial(link ='logit'), data = bank)
summary(banklogit)
anova(banklogit, test="Chisq")

pred8=predict(banklogit, bank, type='response')
table(bank$y, pred8>0.75)
roc8=prediction(as.numeric(pred8), bank$y)
perf8=performance(roc8, "tpr", "fpr")
plot(perf8, col="blue", lwd=1)
abline(0,1, col="red")
auc8=performance(roc8, "auc")
auc8

pred9=predict(banklogit, banktest, type='response')
table(banktest$y, pred9>0.75)
roc9=prediction(as.numeric(pred9), banktest$y)
perf9=performance(roc9, "tpr", "fpr")
plot(perf9, col="blue", lwd=1)
abline(0,1, col="red")
auc9=performance(roc9, "auc")
auc9


