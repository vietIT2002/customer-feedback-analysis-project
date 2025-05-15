# Automated Customer Feedback Analysis Solution for GenZ Fashion Store

## Problem:

The GenZ fashion store is receiving a large volume of customer feedback daily regarding products and shopping experiences on their website. However, due to the sheer volume and diversity of this feedback, it becomes difficult to aggregate, classify, and analyze these responses. This results in delays in improving service and product quality, affecting the store's operational efficiency.

## Goal: 
Propose a technology solution to automatically analyze customer feedback, saving time, optimizing labor costs, and helping the store make timely and effective decisions for improvement.

---

## Solution

### Method:

To address this problem, the team applies the **BERTopic** model combined with **ChatGPT**.

1. **BERTopic Model**: The semantic embedding model (semantic embeddings) combined with clustering algorithms automatically groups customer feedback based on content and context. This helps detect popular topics within the feedback data without manual intervention.

2. **ChatGPT**: Once the feedback is grouped into specific topics, the team further utilizes ChatGPT to interpret and summarize the meaning, purpose, or key concerns of each feedback group. This helps the store better understand the issues that customers care about, allowing for timely decision-making regarding product and service improvements.

---

## Advantages:
- **Automation**: The BERTopic model and ChatGPT automate the process of analyzing customer feedback, saving time and reducing manual intervention.
  
- **Cost Savings**: Reduces labor costs associated with manual classification and analysis of feedback.
  
- **Effectiveness in Detecting Key Issues**: ChatGPT easily interprets the topics derived from customer feedback, helping the store quickly identify and address issues.

---

## Disadvantages:
- **Dependence on Data Quality**: The effectiveness of the BERTopic model is highly reliant on the quality of the feedback data. If the data is abbreviated, contains spelling errors, or lacks context, the analysis may not be accurate.
  
- **Computational Resource Costs**: Implementing models like BERTopic and ChatGPT requires significant computational resources, which can result in high deployment costs.

---

## Conclusion:
By combining the **BERTopic** model and **ChatGPT** to analyze customer feedback, the GenZ fashion store can automatically group and analyze feedback topics efficiently. This method not only saves time and costs but also helps the store improve product and service quality promptly, meeting customer needs quickly and accurately.


