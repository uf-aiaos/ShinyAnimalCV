## ================== ##
##    ShinyAnimalCV   ##
## ================== ##

# Check if required packages are installed, and install if necessary
required_packages <- c("akima", "magick", "reticulate", "shiny", "countcolors", "plotly")
missing_packages <- required_packages[!(required_packages %in% installed.packages())]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

# Load required packages
library(akima)
library(magick)
library(reticulate)
library(shiny)
library(countcolors)

# set up reticulate env (for my mac studio)
reticulate::use_condaenv(condaenv="maskrcnn",conda="auto", required=TRUE)

# Import python functions to R
source("pyfunc.R") 

# load pre-defined functions
getheatmapimg <- function(depthfile) {
  # create heatmap image
  py$plt$cla()
  fig <- py$plt$gcf()
  # specifically set up the image resoluation
  fig$set_size_inches(dim(depthfile)[2]/100, 
                      dim(depthfile)[1]/100,
                      forward=TRUE)
  fig$set_dpi(100)
  py$plt$axis('off')
  py$plt$imshow(depthfile, cmap = "magma", vmin= 0, vmax = 3)
  # convert heatmap img to np array for maskrcnn detection
  fig$canvas$draw()
  depthimg <- py$np$frombuffer(fig$canvas$tostring_rgb(), dtype=py$np$uint8)
  depthimg <- array_reshape(depthimg, append(rev(fig$canvas$get_width_height()), 3))
  return(depthimg)
}

getpretrainedmodel <- function(selectedweight) {
  # load pre-trained weights based on `input$objdtweights`
  weightpath  <- paste0(py$ROOT_DIR, "/logs/", selectedweight, ".h5") 
  model       <- py$modellib$MaskRCNN(mode="inference", 
                                      config=py$inference_config, 
                                      model_dir=py$ROOT_DIR)
  py$tf$keras$Model$load_weights(model$keras_model, weightpath, by_name = TRUE)
  return(model)
}

getobjdtimage <- function(inputimg, morphdf, r) {
  image_data <- py$visualize$save_instances(inputimg, 
                                                  morphdf$minbox, 
                                                  as.list(morphdf$rangle),
                                                  r$masks, 
                                                  r$class_ids,
                                                  list("BG", "ROI"), 
                                                  r$scores) 
  image_data[, , 1:3] <- image_data[, , 1:3] / 255
  return(image_data)
}
# Choose the size of shiny app
options(shiny.maxRequestSize = 100*1024^2)

# Define the Server part for application
server <- function(input, output, session) {
  
  #########################################
  #-- Object segmentation and detection --#
  #########################################
  # reactive variable for input image
  inputimg   <- reactive({
    req(input$objdtfiles)
    imgBGR <- py$cv2$imread(input$objdtfiles$datapath)
    imgRGB <- imgBGR[, , c(3, 2, 1)]
  })
  # reactive variable for a selected pre-trained model
  objdtmodel <- reactive({
    req(input$objdtweights)
    getpretrainedmodel(input$objdtweights)
  })
  # reset output when input file/model changed
  observeEvent(c(input$objdtfiles, input$objdtweights, input$objdtPPM), {
    output$objdtopimg <- renderPlot(NULL)
    output$mfeatureTwoD <- renderTable(NULL)
  })
  # use action button to trigger the inference model and save output
  observeEvent(input$runDet2D, {
    # inference/detection
    res           <- objdtmodel()$detect(list(inputimg()), verbose = 1)
    r             <- res[[1]]
    # calculate mt features
    morphdf       <- py$visualize$morphologicalfeatures(r$masks, r$class_ids, 
                                                        list("BG", "ROI"), 
                                                        ppm = input$objdtPPM)
    dtimgshinyanimalcv    <- getobjdtimage(inputimg(), morphdf, r)
    # render detection image
    output$objdtopimg <- renderPlot({
      plotArrayAsImage(dtimgshinyanimalcv)
    })
    # reder morphological traits
    output$mfeatureTwoD <- renderTable({
      morphdf$morpholdf
    })
    # download detected images
    output$dtimgmaskrcnn <- downloadHandler(
      filename = function() {
        paste("ObjDetection", Sys.Date(), ".png", sep="")
      },
      content = function(file) {
        png(file)
        plotArrayAsImage(dtimgshinyanimalcv)
        dev.off()
      })
    # download morphological data frame
    output$savemfeaturesTwoD <- downloadHandler(
      filename = function() {
        paste("MorpholTrait", Sys.Date(), ".csv", sep = "")
      },
      content = function(file) {
        write.csv(morphdf$morpholdf, file, row.names = FALSE)
      }
    )
  })
  
  ##################################
  #-- 3D surface visualization -- # 
  ##################################
  # reactive variable for input depth file 
  depthfile <- reactive({
    req(input$depthfile)
    filestr   <- input$depthfile
    depthfile <- read.csv(filestr$datapath, header = FALSE)
    return(depthfile)
  })
  # reactive variable for a selected pre-trained model 
  threeDmodel <- reactive({
    req(input$depthweights)
    getpretrainedmodel(input$depthweights)
  })
  # reset output when input file/model changed
  observeEvent(c(input$depthfile, input$depthweights, input$ThreeDPPM, 
                 input$distCtoG, input$sigmaGaussian), {
    output$heatmapimg     <- renderPlot(NULL)
    output$surface3d      <- renderPlotly(NULL)
    output$mfeatureThreeD <- renderTable(NULL)
  })
  observeEvent(input$runDtThreeD, {
    # convert file to heatmap image 
    depthimg      <- getheatmapimg(depthfile())
    # inference/detection
    res           <- threeDmodel()$detect(list(depthimg), verbose = 1)
    r             <- res[[1]]
    depthdf       <- py$visualize$getcleaneddepthdf(depthfile(), r$masks[, , 1], 
                                    distCtoG = input$distCtoG, 
                                    sigma_gaussianfilter = input$sigmaGaussian)
    # calculate mt features
    morphdf       <- py$visualize$morphologicalfeatures(r$masks, r$class_ids, 
                                                        list("BG", "ROI"), 
                                                        ppm = input$ThreeDPPM,
                                                        depthdf = depthdf)
    
    dtheatmapimgshinyanimalcv    <- getobjdtimage(depthimg, morphdf, r)
    # render detection heatmap image
    output$heatmapimg <- renderPlot({
      plotArrayAsImage(dtheatmapimgshinyanimalcv)
    })
    # create 3d surface plot
    output$surface3d <- renderPlotly({
      plot_ly(z = depthdf, type = "surface") %>%
        layout(title = list(text = "3D surface (top-view)", y = 0.95), 
               scene = list(xaxis = list("nticks" = 6), yaxis = list("nticks" = 6)))
    })
    
    output$mfeatureThreeD <- renderTable({
      morphdf$morpholdf
    })
    
    # download detected images
    output$dtheatmap <- downloadHandler(
      filename = function() {
        paste("ObjDetection", Sys.Date(), ".png", sep="")
      },
      content = function(file) {
        png(file)
        plotArrayAsImage(dtheatmapimgshinyanimalcv)
        dev.off()
      })
    # download morphological data frame
    output$savemfeaturesThreeD <- downloadHandler(
      filename = function() {
        paste("MorpholTrait", Sys.Date(), ".csv", sep = "")
      },
      content = function(file) {
        write.csv(morphdf$morpholdf, file, row.names = FALSE)
      }
    )
  })
  
}



