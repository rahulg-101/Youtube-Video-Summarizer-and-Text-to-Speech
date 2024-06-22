import sys

def error_msg_details(error,error_details:sys):
    _,_,exc_tb = error_details.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = f"Error occured, python script name {file_name} at line numer {exc_tb.tb_lineno} & error message {str(error)}"

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details):
        """
        :param error_message : error message in str format
        """
        super().__init__(error_message, self)
        self.error_message = error_msg_details(error_message, error_details=error_details)


    def __str__(self) -> str:
        return self.error_message