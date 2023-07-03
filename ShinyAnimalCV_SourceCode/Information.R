column(8,
       tags$body(
         h3(strong('General information')),
         #br(),
         tags$div(
           tags$p('ShinyAnimalCV is an open-source online web application that designed for performing object detection and segmentation, 
           3D visualization, and extraction of 2D and 3D morphological traits using animal image data. 
           This application aims to facilitate the research and teaching of computer vision within the animal science community. 
           We provide comprehensive guidelines for utilizing this application, both online with pre-trained models and locally for training and utilizing custom data.
           For more information, please refer to the online documentation:', 
           tags$a(href = "https://github.com/uf-aiaos/ShinyAnimalCV", "ShinyAnimalCV Doc", 
                  style = "color:#0096FF")), 
           style = "font-size: 18px;"),
         h3(strong('Contact Information and Help:')),
         tags$div(
           style = "display: flex; align-items: flex-start;",
           tags$ul(
             style = "list-style-type: disc; margin-right: 10px; padding-left: 20px;",
             tags$li(
               tags$p(
                 strong('Jin Wang'),
                 tags$a(
                   class = "shiny__link",
                   href = "mailto:jin.wang@ufl.edu",
                   style = "color:#6AB1EC;",
                   strong('(jin.wang@ufl.edu)')
                 )
               )
             ),
             tags$li(
               tags$p(
                 strong('Haipeng Yu'),
                 tags$a(
                   class = "shiny__link",
                   href = "mailto:haipengyu@ufl.edu",
                   style = "color:#6AB1EC;",
                   strong('(haipengyu@ufl.edu)')
                 )
               )
             )
           )
         ),
      br(),
      tags$div(
        tags$footer('Copyright (C) 2023, code licensed under GPLv3'),
        style = "font-size: 16px")
))




         
       
      
       

       
      
