"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.appmetadata import AppMetadata

APP_VERSION = 0.1
# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # Basic Metadata
    metadata = AppMetadata(
        name="Slate Text Detection",
        description="This tool applies a custom Faster-RCNN model to slates specified in the input mmif.",  
        app_license="MIT",
        identifier="slate-textdetection",
        url="https://github.com/clamsproject/app-slate-textdetection", 
    )
    
    # IO Spec
    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_input(AnnotationTypes.TimeFrame, required=True, frameType='slate')
    
    metadata.add_output(AnnotationTypes.BoundingBox)
    return metadata 


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    sys.stdout.write(appmetadata().jsonify(pretty=True))
