## ================== ##
##    ShinyAnimalCV   ##
## ================== ##
library(plotly)
# Define UI for the application
ui <- function(req){
  fluidPage(
    title = "ShinyAnimalCV", # Webpage title
    # Create the Application Title uing headerPanel
    headerPanel(h1("ShinyAnimalCV: open-source cloud-based web application for object detection and segmentation and three-dimensional visualization using computer vision",
                   style = "font-family: 'Impact', fantasy;
                   font-weight: 500; line-height: 1.1;
                   color: #6AB1EC;", align = "center")),
    # Blocks printing any errors in the Shiny UI.
    tags$style(type="text/css",
               ".shiny-output-error { visibility: hidden; }",
               ".shiny-output-error:before { visibility: hidden; }"),
    # CSS for checkbox style
    tags$style("
        .checkbox { /* checkbox is a div class*/
          line-height: 38px;
          margin-bottom: 30px; /*set the margin, so boxes don't overlap*/
        }
        input[type='checkbox']{ /* style for checkboxes */
          width: 30px; /*Desired width*/
          height: 50px; /*Desired height*/
          line-height: 30px;
        }
        span {
            margin-left: 15px;  /*set the margin, so boxes don't overlap labels*/
            line-height: 30px;
        }
    "),
    
    # Create tabset and tabPanels for the application
    tabsetPanel(
      # First tabpanel is the "Main window" and all the information for application can be found in Information.R file
      tabPanel(
        h4("Information", style = "color: #000000;font-weight: bold"),
        br(),
        source("Information.R", local = TRUE)[1]
      ),
      # #============================Object Detection=================================# 
      tabPanel(
        h4("Object detection and segmentation", style = "color: #0096FF;font-weight: bold"),
        br(),
        sidebarLayout(
          sidebarPanel(
            # h4("Single image detection", style = "color: #000000;font-weight: bold"),
            # br(),
            fileInput(inputId = 'objdtfiles', 
                      label = 'Step 1: Upload an image in RGB color model',
                      multiple = TRUE),
            selectInput(inputId = 'objdtweights', 
                        label = 'Step 2: Select a pretrained machine learning model',
                        list("",
                             `Pig` = list("OnePig_OriginalRGB", "OnePig_DepthRGB", "OnePig_HeatmapRGB",
                                          "TwoPigs_OriginalRGB", "TwoPigs_DepthRGB", "TwoPigs_HeatmapRGB","FourPigs_OriginalRGB"), 
                             `DairyCattle` = list("OneCattle_OriginalRGB","OneCattle_DepthRGB", "OneCattle_HeatmapRGB"),
                             `Custom model`= list("CustomWeights") # note: this Custom model is only available for local deployment
                             )), 
            tags$div(style = "margin-bottom: 2px;"),
            HTML("To train and utilize weights based on custom data, 
                 please refer to the online documentation: 
                 <b><a href='https://github.com/uf-aiaos/ShinyAnimalCV' style='color: #0096FF;'>ShinyAnimalCV Doc</a></b>."),
            tags$div(style = "margin-bottom: 20px;"),
            numericInput(inputId = 'objdtPPM',
                         label = 'Step 3: Enter the pixels per meter (e.g., image width in pixels / width of field of view in meter)',
                         1, min = 1, max = NA),
            fluidRow(column(12, align = "left",
                            strong("Step 4:"),
                            br(),
                            actionButton(inputId = 'runDet2D', label = strong('Run the computer vision model'), 
                                         icon = icon("arrows-rotate"), style = "color: #0096FF;"))),
            br(),
            h5("Save output:", style = "font-weight: bold;"),
            fluidRow(column(12, align = "left",
                            downloadButton("dtimgmaskrcnn", "Save the detected and segmented image", width = '150%',
                                           style = "color: #0096FF;"))),
            br(),
            fluidRow(column(12, align = "left",
                            downloadButton("savemfeaturesTwoD", "Save 2D morphological features", width = '150%',
                                           style = "color: #0096FF;"))),
          ),
          mainPanel(
            tabsetPanel(
              tabPanel(strong("Results"),  
                       column(10, strong("Detected and segmented image", style = "color: #0096FF;"),  plotOutput("objdtopimg")),
                       column(10, strong("2D morphological features", style = "color: #0096FF;"), tableOutput("mfeatureTwoD"))
                       )
            )
            
        )
      )
    ),
      #========================== 3D surface visualization ============================#
      tabPanel(
        h4("3D morphological feature extraction and visualization", style = "color: #000000;font-weight: bold"),
        br(),
        sidebarLayout(
          sidebarPanel(
            fileInput(inputId = 'depthfile', 
                      label = 'Step 1: Upload a csv file with depth information',
                      multiple = FALSE),
            selectInput(inputId = 'depthweights',
                        label = 'Step 2: Select a pretrained machine learning model',
                        list("",
                          `Pig` = list("OnePig_HeatmapRGB"),
                          `DairyCattle` = list("OneCattle_HeatmapRGB"),
                          `Custom model`= list("CustomWeights") # note: this Custom model is only available for local deployment
                          ),selected = NULL, multiple = FALSE),
            tags$div(style = "margin-bottom: 2px;"),
            HTML("To train and utilize weights based on custom data,
                  please refer to the online documentation: 
                 <b><a href='https://github.com/uf-aiaos/ShinyAnimalCV' style='color: #0096FF;'>ShinyAnimalCV Doc</a></b>."),
            tags$div(style = "margin-bottom: 20px;"),
            numericInput(inputId = 'ThreeDPPM',
                         label = 'Step 3: Enter the pixels per meter (e.g., image width in pixels / width of field of view in meter)',
                         1, min = 1, max = NA),
            numericInput(inputId = 'distCtoG',
                        label = 'Step 4: Enter the distance from camera to ground in meter (enter 0 to estimate this distance from the depth data)',
                        2.5, min = 0, max = 100),
            numericInput(inputId = 'sigmaGaussian',
                         label = 'Step 5: Enter the standard deviation for Gaussian filter (this value controls the amount of smoothing applied to 3D surface, with larger value resulting in more smoothing)',
                         5, min = 0, max = 100),
            fluidRow(column(12, align = "left",
                            strong("Step 6:"),
                            br(),
                            actionButton(inputId = 'runDtThreeD', label = strong('Run the computer vision model'), 
                                         icon = icon("arrows-rotate"), style = "color: #0096FF;"))),
            br(),
            h5("Save output:", style = "font-weight: bold;"),
            fluidRow(column(12, align = "left",
                            downloadButton("dtheatmap", "Save the detected and segmented image", width = '150%',
                                           style = "color: #0096FF;"))),
            br(),
            fluidRow(column(12, align = "left",
                            downloadButton("savemfeaturesThreeD", "Save 3D morphological features", width = '150%',
                                           style = "color: #0096FF;"))),
          ),
          mainPanel(
            tabsetPanel(
              tabPanel(strong("Results"),
                       fluidRow(
                         column(5, strong("Detected and segmented image", style = "color: #0096FF;"),
                                plotOutput("heatmapimg")),
                         column(7, strong("3D visualization", style = "color: #0096FF;"),
                                plotlyOutput("surface3d"))
                         ),
                       fluidRow(
                         column(10, strong("3D morphological features", style = "color: #0096FF;"), tableOutput("mfeatureThreeD"))
                       )
                       ),
            )
          )
          )
        ),
    ) # end of tabsetPanel
  )   # end of fluidPage
}     # end of function

      
    
