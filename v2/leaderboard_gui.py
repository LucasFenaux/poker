import tkinter as tk
from tkinter import ttk
import ray


class LeaderboardGUI:
    def __init__(self, actor_handle):
        self.actor = actor_handle
        self.root = tk.Tk()
        self.root.title("Casino Live Leaderboard")
        self.root.geometry("600x400")  # Made it slightly wider to fit the new column

        # UI Elements
        self.label = tk.Label(self.root, text="Live Standings", font=("Arial", 16, "bold"))
        self.label.pack(pady=10)

        # Added "Games" to the columns list
        self.tree = ttk.Treeview(self.root, columns=("Rank", "ID", "Games", "Total", "Recent Avg"), show='headings')
        self.tree.heading("Rank", text="Rank")
        self.tree.heading("ID", text="Player ID")
        self.tree.heading("Games", text="Games")  # New heading
        self.tree.heading("Total", text="All-Time")
        self.tree.heading("Recent Avg", text="Last 10 Avg")

        self.tree.column("Rank", width=50)
        self.tree.column("Games", width=60)  # Set width for new column
        self.tree.pack(expand=True, fill='both', padx=10, pady=10)

        # Start the update loop
        self.refresh_data()
        self.root.mainloop()

    def refresh_data(self):
        try:
            # Fetch data from the actor
            data = ray.get(self.actor.get_leaderboard_stats.remote())

            # Clear old entries
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Insert new entries including the game_count
            for i, p in enumerate(data['all_time'], 1):
                self.tree.insert("", "end", values=(
                    i,
                    p['id'],
                    p['game_count'],  # Insert the individual player's game count
                    f"{p['total']:.2f}",
                    f"{p['recent_avg']:.2f}"
                ))

            # Update the label using the overall avg_games
            if data.get('is_done'):
                self.label.config(text=f"Final Standings (Avg Games: {data['avg_games']:.1f}) - FINISHED")
                self.root.after(1000, self.root.destroy)
                return
            else:
                self.label.config(text=f"Live Standings (Avg Games: {data['avg_games']:.1f})")

        except Exception as e:
            print(f"Waiting for actor... {e}")

        # Schedule next update in 2000ms (2 seconds)
        self.root.after(2000, self.refresh_data)


if __name__ == "__main__":
    print("Connecting to Ray cluster inside Docker...")

    # Connect to the Ray cluster running in Docker over port 10001
    # Note: ray:// is required for Ray Client connections
    ray.init(address="ray://localhost:10001", namespace="casino")
    # We need a way to find the LeaderboardActor that the CasinoManager created.
    # Ray allows us to fetch actors by their assigned name.
    try:
        # We will need to name the actor in the backend first (see Step 3)
        leaderboard_actor = ray.get_actor("GlobalLeaderboard")
        print("Successfully connected to the Leaderboard!")

        # Start the GUI
        gui = LeaderboardGUI(leaderboard_actor)

    except ValueError:
        print("Could not find the LeaderboardActor. Is the Casino running in Docker?")
    finally:
        ray.shutdown()