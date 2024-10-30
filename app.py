import boto3
import botocore.config
import json
from datetime import datetime


def blog_generate_using_bedrock(blogtopic: str) -> str:
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Write a 200-word blog on the topic {blogtopic}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }


    try:
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1",
                               config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3}))
        response = bedrock.invoke_model(body=json.dumps(body), modelId="meta.llama3-8b-instruct-v1:0")

        response_content = response.get('body').read()
        response_data = json.loads(response_content)
        print(response_data)
        blog_details = response_data['generation']
        return blog_details
    except Exception as e:
        print(f"Error generating the blog: {e}")
        return ""
    
   ##**def lambda_handler(event, context):
    #event = json.loads(event['body'])
    #blogtopic = event['blog_topic']

    


    

   



