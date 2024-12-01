import praw
import csv

# Reddit API credentials
reddit = praw.Reddit(
    client_id="by4YpcKlxE2_ZTdJHnjC6g",
    client_secret="4nSVzuL9w36MS08vWFngcZXpkAJE0g",
    user_agent="MentalHealthDataCollector/1.0 (by u/PhilosophySecure109)",
    redirect_uri="http://localhost:8080"
)

# Specify subreddit
subreddit = reddit.subreddit("Anxiety")  # Change to "anxiety" or others as needed

# Open a CSV file for writing
with open("reddit_posts_comments_Anxiety.csv", "w", newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    # Write the header row
    writer.writerow(["Post Title", "Post Body", "Comment"])
    
    # Fetch posts
    for submission in subreddit.hot(limit=600):  # Change limit as needed
        print(f"Fetching post: {submission.title}")
        submission.comments.replace_more(limit=20)  # Expand all comments

        # Loop through all comments and write them to CSV
        for comment in submission.comments.list():
            # Get full comment text
            comment_text = comment.body.strip()

            # Avoid empty comments (which can sometimes appear as blank lines)
            if comment_text:
                writer.writerow([submission.title, submission.selftext, comment_text])
