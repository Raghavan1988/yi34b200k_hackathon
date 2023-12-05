# yi34b200k_hackathon
Lablab ai hackathon submission, using Yi 34 B to extend the ideas of an ARXIV paper and apply it to new field


## Requirements
- Python environment
- Access to Yi-34 B model (token provided for judges)
- `replicate` Python package
- Arxiv paper URL and the target field for application
- Arxiv data with metadata json file (since 12/2022 to 08/2023)
-   https://arxiv-r-1228.s3.us-west-1.amazonaws.com/arxiv_12_28_2022.json
-   https://arxiv-r-1228.s3.us-west-1.amazonaws.com/annoy_index_since_dec22.ann

## Steps

### 1. Setting Up the Environment
- Clone the GitHub repository: [yi34b200k_hackathon](https://github.com/Raghavan1988/yi34b200k_hackathon).
- Install required packages:
  ```bash
  pip install -r requirements.txt
  export FLASK_APP=arxiv_summarizer.py
  export REPLICATE_API_TOKEN= <your token>
  flask run
![Screenshot from 2023-12-04 23-38-09](https://github.com/Raghavan1988/yi34b200k_hackathon/assets/493090/08d58e6e-11b0-49f9-a50e-a07f4416270b)
