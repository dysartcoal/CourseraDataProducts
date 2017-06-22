# 
# This is the server for the week4dp application.  The application supports spot 
# checking of various machine learning algorithms against binary classification data by
# providing box plots of Accuracy and Kappa measures, data summaries and confusion matrices.
#
# The server loads the data, runs the 10 machine learning methods against the data and 
# builds the ggplot objects for the various confusion matrices prior to 
# presenting the interactive interface to the user.  The machine learning methods are not 
# run after the initialisation of the app.
#
# The multiplot function is defined to enable plotting of lists of ggplot objects.  This is 
# used to present the confusion matrices for all of the selected ML methods.
#
# The load_data function supports the delayed presentation of the interface to the user by
# showing the "Loading..." message until the underlying models are built and ready to be used.
#
# The call to the server function displays the boxplots, data summary and confusion 
# matrices for the Machine Learning methods selected in the user interface.  Changes
# to the checkboxes selected results in an update to the plots and the summary data.
#

library(shiny)
library(shinyjs) 
library(dplyr)
library(caret)
library(ggplot2)
library(glmnet)
library(Matrix)
library(kernlab)
library(foreach)
library(randomForest)
library(ipred)
library(e1071)
library(C50)
library(plyr)
library(rpart)
library(MASS)
library(klaR)


#
# Code for the multiplot function obtained from:
# http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_%28ggplot2%29/
#
#
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
    require(grid)
    
    # Make a list from the ... arguments and plotlist
    plots <- c(list(...), plotlist)
    
    numPlots = length(plots)
    
    # If layout is NULL, then use 'cols' to determine layout
    if (is.null(layout)) {
        # Make the panel
        # ncol: Number of columns of plots
        # nrow: Number of rows needed, calculated from # of cols
        layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                         ncol = cols, nrow = ceiling(numPlots/cols))
    }
    
    if (numPlots==1) {
        print(plots[[1]])
        
    } else {
        # Set up the page
        grid.newpage()
        pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
        
        # Make each plot, in the correct location
        for (i in 1:numPlots) {
            # Get the i,j matrix positions of the regions that contain this subplot
            matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
            
            print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                            layout.pos.col = matchidx$col))
        }
    }
}


# Load data function
#
# Returns a list of the models so is dependent on the models having been built.
# On completion of building the model_list the loading page is hidden in the UI.
#
load_data <- function() {
    model_list <- list(lda=fit.lda, glm=fit.glm, glmnet=fit.glmnet,
                       svmRadial=fit.svmRadial, knn=fit.knn, nb=fit.nb, 
                       rpart=fit.cart, c50=fit.c50,
                       treebag=fit.treebag, rf=fit.rf)
    
    hide("loading_page")
    show("main_content")
    model_list
}



# Initialise server
#
# Initialise the server by reading in the Vertebral Column Data Set and building the 
# the 10 models using the different methods passed as arguments to the caret train function.
#
# ggplot objects are created to display 3 confusion matrices of counts, 
# sensitivity/specificity and positive and negative predictive values for each of the 
# models when run against the data set.  This provides an in-sample indication of 
# performance.

#
#   Initialise the width so that summary data does not wrap in the call to outputVerbatimText.
#
options(width = 200) 


#
#  Load data
#

sp <- read.csv("spinal.csv")


#
# Train models
#

seed <- 1313
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"

# Naive Bayes
set.seed(seed)
fit.nb <- train(result ~ ., data=sp, method="nb", metric=metric, 
                trControl=control)
# CART
set.seed(seed)
fit.cart <- train(result ~ ., data=sp, method="rpart", metric=metric, 
                  trControl=control)
# C5.0
set.seed(seed)
fit.c50 <- train(result ~ ., data=sp, method="C5.0", metric=metric, 
                 trControl=control)
# Bagged CART
set.seed(seed)
fit.treebag <- train(result ~ ., data=sp, method="treebag", metric=metric, 
                     trControl=control)
# Random Forest
set.seed(seed)
fit.rf <- train(result ~ ., data=sp, method="rf", metric=metric, 
                trControl=control)
# Linear Discriminant Analysis
set.seed(seed)
fit.lda <- train(result ~ ., data=sp, method="lda", metric=metric, 
                 preProc=c("center", "scale"), trControl=control)
# Logistic Regression
set.seed(seed)
fit.glm <- train(result ~ ., data=sp, method="glm", metric=metric, 
                 trControl=control)
# GLMNET
set.seed(seed)
fit.glmnet <- train(result ~ ., data=sp, method="glmnet", metric=metric, 
                    preProc=c("center", "scale"), trControl=control)
# SVM Radial
set.seed(seed)
fit.svmRadial <- train(result ~ ., data=sp, method="svmRadial", metric=metric, 
                       preProc=c("center", "scale"), trControl=control, fit=FALSE)
# kNN
set.seed(seed)
fit.knn <- train(result ~ ., data=sp, method="knn", metric=metric, 
                 preProc=c("center", "scale"), trControl=control)



#
# Summarise data
#
# Build a dataframe for each model that contains the reference and predicted classes
# to enable confusion matrix plots.  The plots are added to a list so that they can 
# be selected or deselected in line with the user selection.
#


TClass <- factor(c("Pos", "Pos", "Neg", "Neg"), levels=c("Pos", "Neg"))
PClass <- factor(c("Neg","Pos","Neg","Pos"), levels=c("Neg", "Pos"))

plotvals <- list()
plotsenspec <- list()
plotpospred <- list()
models <- c("lda", "glm", "glmnet",
            "svmRadial", "knn", "nb", 
            "rpart", "c50", "treebag",
            "rf")

#
#  Build the ggplot objects for each of the confusion matrices for each model.
#  This should have been done as a list operation not a for loop.  
#
for(i in models) {
    title <- i
    if(i=="lda") fit.model <- fit.lda
    if(i=="glm") fit.model <- fit.glm
    if(i=="glmnet") fit.model <- fit.glmnet
    if(i=="svmRadial") fit.model <- fit.svmRadial
    if(i=="knn") fit.model <- fit.nb
    if(i=="nb") fit.model <- fit.lda
    if(i=="rpart") fit.model <- fit.cart
    if(i=="c50") fit.model <- fit.c50
    if(i=="treebag") fit.model <- fit.treebag
    if(i=="rf") fit.model <- fit.rf
    
    pred <- predict(fit.model, newdata=sp)
    cm <- confusionMatrix(sp$result, pred)
    Y <- c(cm$table[2,1], cm$table[1,1], cm$table[2,2], cm$table[1,2])
    SensSpec <- c(Y[1]/(Y[1] + Y[2]), Y[2]/(Y[1] + Y[2]), 
                  Y[3]/(Y[3] + Y[4]), Y[4]/(Y[3] + Y[4]))
    PosPredNegPred <- c(Y[1]/(Y[1] + Y[3]), Y[2]/(Y[2] + Y[4]), 
                        Y[3]/(Y[1] + Y[3]), Y[4]/(Y[2] + Y[4]))
    df <- data.frame(TClass, PClass, Y, SensSpec, PosPredNegPred, 
                     model=c(title, title, title, title))
    models[i] <- title
    plotvals[[i]] <- ggplot(data=df, mapping=aes(x=TClass, y=PClass)) + 
        geom_tile(aes(fill=Y), colour="white") + 
        geom_text(aes(label=sprintf("%1.0f", Y)), vjust=1) + 
        scale_fill_gradient(low="white", high="red") + 
        theme_bw() + 
        theme(legend.position="none") + 
        xlab("Reference Class") + 
        ylab("Predicted Class") + 
        ggtitle(title)
    plotsenspec[[i]] <- ggplot(data=df, mapping=aes(x=TClass, y=PClass)) + 
        geom_tile(aes(fill=SensSpec), colour="white") + 
        geom_text(aes(label=sprintf("%1.3f", SensSpec)), vjust=1) + 
        scale_fill_gradient(low="white", high="red") + 
        theme_bw() + 
        theme(legend.position="none") + 
        xlab("Reference Class") + 
        ylab("Predicted Class") + 
        ggtitle(title)
    plotpospred[[i]] <- ggplot(data=df, mapping=aes(x=TClass, y=PClass)) + 
        geom_tile(aes(fill=PosPredNegPred), colour="white") + 
        geom_text(aes(label=sprintf("%1.3f", PosPredNegPred)), vjust=1) + 
        scale_fill_gradient(low="white", high="red") + 
        theme_bw() + 
        theme(legend.position="none") + 
        xlab("Reference Class") + 
        ylab("Predicted Class") + 
        ggtitle(title)
}




#
# shinyServer
#

shinyServer(function(input, output, session) {
    model_list <- load_data()
    
    #
    # check() reactive variable is created to control output when less than 2 
    # ML methods are selected by the user.  both the boxplot and the 
    # data summary require at least 2 methods to be selected.
    #
    # results() reactive variable is created to support the results of resampling
    # with the selected ML methods.  The results() data is the basis of the
    # boxplot and data summary outputs.
    #
    check <- reactive({length(input$method)})
    results <- reactive({
        resamples(model_list[input$method])
    })
    
    
    #
    # Output boxplot
    #
    
    output$bwplot <- renderPlot({
        if (check() > 1) {
            bwplot(results())
        } else {
            plot(c(), c(), xlim=c(0, 2), ylim=c(0,2))
            text(1,1, "Please select at least 2 methods.")
        }
    })
    
    
    #
    # Output data summary 
    #
    
    output$dataSummary <- renderPrint({
        if (check() > 1) {
            summary(results())
        } else {
            "Please select at least 2 methods."
        }
    })
    
    
    #
    # Output confusion matrix with counts
    #
    
    output$cmvals <- renderPlot({
       ind <- which(models %in% input$method) 
       multiplot(plotlist=plotvals[ind], cols=2)
    })
    
    # renderUI is used to enable the height of the plots to be specified
    output$cmvals.ui <- renderUI({
        my_height <- 200*check()
        plotOutput("cmvals", height=my_height)
    })
    
    
    #
    # Output confusion matrix with Sensitivity and Specificity
    #
    
    output$cmsenspec <- renderPlot({
        ind <- which(models %in% input$method) 
        multiplot(plotlist=plotsenspec[ind], cols=2)
    })
    
    # renderUI is used to enable the height of the plots to be specified
    output$cmsenspec.ui <- renderUI({
        my_height <- 200*check()
        plotOutput("cmsenspec", height=my_height)
    })
    
    
    #
    # Output confusion matrix with positive and negative predictive values
    #
    
    output$cmpospred <- renderPlot({
        ind <- which(models %in% input$method) 
        multiplot(plotlist=plotpospred[ind], cols=2)
    })
    
    # renderUI is used to enable the height of the plots to be specified
    output$cmpospred.ui <- renderUI({
        my_height <- 200*check()
        plotOutput("cmpospred", height=my_height)
    })
})


