# 
# This is the user interface for the week4dp application.  The application supports spot 
# checking of various machine learning algorithms against binary classification data by
# providing box plots of Accuracy and Kappa measures, data summaries and confusion matrices.
#
# The interface contains 3 main sections:
#   - title and introductory text
#   - input panel with a list of checkboxes against machine learning methods
#   - tab panel with tabs for 
#       o help
#       o boxplots
#       o accuracy and kappa summary data
#       o confusion matrix displaying values
#       o confusion matrix displaying sensitivity and specificity
#       o confusion matrix displaying positive and negative predictive values
#
#


library(shiny)
library(shinyjs)

shinyUI(fluidPage(
    useShinyjs(),
    div(
        id = "loading_page",
        h1("Loading...")
    ),
    div(
        id = "main_content",
        
        # Application title
        titlePanel("Binary Classifier - Spot Check"),
        
        #
        #  The introduction
        #
        
        fluidRow(
            p("Selecting an appropriate machine learning algorithm to best predict binary classes from data is not a deterministic process.  Quickly and roughly testing the data or a subset of the data with a range of different machine learning methods may give an indication of which algorithms are worth pursuing in greater detail.  This application provides summary plots and data to support decision making and is based on an article by Machine Learning Mastery called ", a(href="http://machinelearningmastery.com/evaluate-machine-learning-algorithms-with-r/", "How to Evaluate Machine Learning Algorithms with R.", target="_blank"))
        ),
        # 
        fluidRow(
            column(3, 
                   
                   #
                   #  The input checkboxes
                   #
                   
                   inputPanel(
                       checkboxGroupInput("method", "Show ML Method:", 
                                          c("lda" = "lda", "glm" = "glm", 
                                            "glmnet" = "glmnet", 
                                            "svmRadial" = "svmRadial", "knn" = "knn", 
                                            "nb" = "nb", "rpart" = "rpart",
                                            "C5.0" = "c50", "treebag" = "treebag", 
                                            "rf" = "rf"), 
                                          selected=c( "nb", 
                                                     "rpart"))
                       
                   )
            ),
            column(9,
                   
                   #
                   #  The tabs
                   #
                   
                   tabsetPanel(
                       tabPanel("Help", 
                                tags$h4("Initialising the App"),
                                p("All of the machine learning models are built up front and can take a few minutes to complete. While this is happening the word 'Loading...' appears in the top left hand corner of the window.  Please be patient during this computation."),
                                tags$h4("Using the App"),
                                p("Select and deselect the machine learning methods using the checkboxes on the left. Select the required tab above to view the Boxplot, Confusion Matrix, Sensitivity and Specificity or Predictive Values for the selected Machine Learning methods."),
                                tags$h4("Description"),
                                p("In this application the data that is used has been downloaded from the UCI Machine Learning Repository and is called the Vertebral Column Data Set. This biomedical data set was built by Dr. Henrique da Mota and consists of classifying patients as belonging to one out of two categories: Normal (100 patients) or Abnormal (210 patients).  More information can be found ", a(href="https://archive.ics.uci.edu/ml/datasets/Vertebral+Column", "here.", target="_blank")),
                                p("Although the Vertebral Column Data Set is used for this demonstration, the application will work with any binary classification data set which can be loaded into a dataframe where the outcome field is labelled 'result'."),
                                p("The caret package in R is used to build the models according to the 10 methods listed against the checkboxes to the left. Resampling is used to establish the in-sample Accuracy and Kappa data for each of the models while the confusion matrices, sensitivity/specificity and predictive values are calculated from a single in-sample prediction of outcomes.")
                                ),
                       
                       #
                       #  The data and plot tabs
                       #
                       
                       tabPanel("Boxplot", plotOutput("bwplot")),
                       tabPanel("Summary", verbatimTextOutput("dataSummary")),
                       tabPanel("Confusion Matrices",  uiOutput("cmvals.ui")),
                       tabPanel("Sensitivity/Specificity",  uiOutput("cmsenspec.ui")),
                       tabPanel("Predictive Vals",  uiOutput("cmpospred.ui"))
                   )
            )
        )
    )
)
)
