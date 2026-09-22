
#read the curated resource file
import json
# handle the resource file path safely
from pathlib import Path


# loads the curated resource collection from a json file
def load_resources(file_path):
    
    file_path = Path(file_path)
    # return an empty result when the resource file is unavailable
    if not file_path.exists():
        return []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        resources = json.load(file)
    return resources


# finds one resource by its stored resource id
def get_resource_by_id(resources, resource_id):
    for resource in resources:
        if resource.get("id") == resource_id:
            return resource

    # no matching resource found
    return None


# return the resources that match a requested topic
def get_resources_by_topic(resources, topic):
   
    # normalise the requested topic before comparing it
    requested_topic = topic.strip().lower()

    matching_resources = []
    for resource in resources:
        topics = resource.get("topics", [])
        # normalise the stored topic names as well
        normalised_topics = [
            item.strip().lower()
            for item in topics
        ]

        if requested_topic in normalised_topics:
            matching_resources.append(resource)
    return matching_resources