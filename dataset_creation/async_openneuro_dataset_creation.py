import time
import asyncio
import json
from aiobotocore.session import get_session

class IncrementalEncoder(json.JSONEncoder):
    def encode(self, o):
        partial_result = ""
        for chunk in super().iterencode(o):
            partial_result += chunk
            yield partial_result
            partial_result = ""

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken by {func.__name__}: {elapsed_time:.8f} seconds")
        return result
    return wrapper

@time_it
async def list_folders(bucket, prefix='', delimiter='/'):
    folder_names = []
    continuation_token = None

    session = get_session()
    async with session.create_client('s3') as s3:
        while True:
            # Include the continuation token in the request if it exists
            kwargs = {'Bucket': bucket, 'Prefix': prefix, 'Delimiter': delimiter}
            if continuation_token:
                kwargs['ContinuationToken'] = continuation_token

            response = await s3.list_objects_v2(**kwargs)
            folder_names.extend([x['Prefix'].split('/')[-2] for x in response.get('CommonPrefixes', [])])

            # Check if more items are available to retrieve
            if 'NextContinuationToken' in response:
                continuation_token = response['NextContinuationToken']
            else:
                break

    return folder_names

@time_it
async def list_objects_in_folder(bucket, folder):
    object_keys = []
    continuation_token = None

    session = get_session()
    async with session.create_client('s3') as s3:
        while True:
            # Include the continuation token in the request if it exists
            kwargs = {'Bucket': bucket, 'Prefix': folder}
            if continuation_token:
                kwargs['ContinuationToken'] = continuation_token

            response = await s3.list_objects_v2(**kwargs)
            object_keys.extend([obj['Key'] for obj in response.get('Contents', [])])

            # Check if more items are available to retrieve
            if 'NextContinuationToken' in response:
                continuation_token = response['NextContinuationToken']
            else:
                break

    return object_keys

@time_it
async def main():
    bucket_name = 'openneuro.org'
    prefix = ''
    json_file_path = 'openneuro_objects.json'

    folders = await list_folders(bucket_name, prefix)
    
    result = {}
    remaining_folders = len(folders)
    
    for folder in folders:
        print(f"Number of folders remaining are {remaining_folders}")
        object_keys = await list_objects_in_folder(bucket_name, folder)        
        result[folder] = object_keys

        # Write the current result to the JSON file
        with open(json_file_path, 'w') as file:
            json.dump(result, file, cls=IncrementalEncoder, indent=2)
            print(f"Partial result saved to {json_file_path}")
        remaining_folders -= 1

if __name__ == "__main__":
    asyncio.run(main())