import tkinter as tk
from tkinter import ttk, messagebox
import torch
import os
import argparse
import glob
import random

# --- Local Project Imports ---
from src.app.play_vs_ai import HumanAIPokerManager
from src.ppo_self_play.alg import PPO, PPOInferenceWrapper


def format_cards(card_list):
    """Converts ['As', 'Th'] into 'A♠ T♥'"""
    if not card_list:
        return ""
    suit_symbols = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    formatted = []
    for card in card_list:
        card = card.strip("'")
        if len(card) >= 2:
            rank, suit = card[0], card[1].lower()
            symbol = suit_symbols.get(suit, suit)
            formatted.append(f"{rank}{symbol}")
        else:
            formatted.append(card)
    return "  ".join(formatted)


class PokerGameGUI:
    def __init__(self, root, manager):
        self.root = root
        self.root.title("Man vs Machine: Heads-Up Poker")
        self.root.geometry("750x700")  # Slightly taller to comfortably fit the street wagers
        self.root.configure(bg="#2E8B57")

        self.manager = manager
        self.create_widgets()

        self.root.bind('<Return>', self.handle_enter_key)
        self.prompt_setup()

    def prompt_setup(self):
        setup_win = tk.Toplevel(self.root)
        setup_win.title("Game Setup")
        setup_win.geometry("320x350")
        setup_win.transient(self.root)
        setup_win.grab_set()
        setup_win.protocol("WM_DELETE_WINDOW", self.root.destroy)

        tk.Label(setup_win, text="Initial Stack:", font=("Arial", 12)).pack(pady=5)
        chips_var = tk.StringVar(value="200")
        tk.Entry(setup_win, textvariable=chips_var, font=("Arial", 12)).pack()

        tk.Label(setup_win, text="Small Blind:", font=("Arial", 12)).pack(pady=5)
        sb_var = tk.StringVar(value="1")
        tk.Entry(setup_win, textvariable=sb_var, font=("Arial", 12)).pack()

        tk.Label(setup_win, text="Big Blind:", font=("Arial", 12)).pack(pady=5)
        bb_var = tk.StringVar(value="2")
        tk.Entry(setup_win, textvariable=bb_var, font=("Arial", 12)).pack()

        # Toggles (Both True by default)
        disable_mucking_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(setup_win, text="Disable AI Mucking (Always show cards)", variable=disable_mucking_var).pack(
            pady=5)

        display_bb_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(setup_win, text="Display values in Big Blinds (BB)", variable=display_bb_var).pack(pady=5)

        def apply_setup(event=None):
            try:
                chips = float(chips_var.get())
                sb = float(sb_var.get())
                bb = float(bb_var.get())
                disable_muck = disable_mucking_var.get()
                disp_bb = display_bb_var.get()

                self.manager.set_game_params(chips, sb, bb, disable_muck, disp_bb)
                setup_win.grab_release()
                setup_win.destroy()

                self.manager.start_new_hand()
                self.update_ui()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values.")

        setup_win.bind('<Return>', apply_setup)

        ttk.Button(setup_win, text="Start Game", command=apply_setup).pack(pady=10)

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12, "bold"), padding=5)

        # 1. AI Area
        self.ai_frame = tk.Frame(self.root, bg="#2E8B57")
        self.ai_frame.pack(pady=10, fill="x")

        self.ai_label = tk.Label(self.ai_frame, text="🤖 AI Player", font=("Arial", 14, "bold"), bg="#2E8B57",
                                 fg="white")
        self.ai_label.pack()
        self.ai_stack_label = tk.Label(self.ai_frame, text="Stack: --", font=("Arial", 12), bg="#2E8B57", fg="white")
        self.ai_stack_label.pack()

        self.ai_cards_label = tk.Label(self.ai_frame, text="[ Hidden ]", font=("Courier", 20, "bold"), bg="#2E8B57",
                                       fg="white")
        self.ai_cards_label.pack(pady=5)

        self.ai_bet_label = tk.Label(self.ai_frame, text="", font=("Arial", 12, "bold"), bg="#2E8B57", fg="#FFD700")
        self.ai_bet_label.pack(pady=2)

        # 2. Board and Pot Area
        self.board_frame = tk.Frame(self.root, bg="#1a4d30", bd=3, relief="ridge")
        self.board_frame.pack(pady=10, padx=50, fill="both", expand=True)

        self.pot_label = tk.Label(self.board_frame, text="Pot: 0", font=("Arial", 16, "bold"), bg="#1a4d30", fg="gold")
        self.pot_label.pack(pady=10)

        self.board_cards_label = tk.Label(self.board_frame, text="[ Board Cards ]", font=("Courier", 24, "bold"),
                                          bg="#1a4d30", fg="white")
        self.board_cards_label.pack(pady=10)

        self.status_label = tk.Label(self.board_frame, text="Configuring Game...", font=("Arial", 14, "italic"),
                                     bg="#1a4d30", fg="#FFD700")
        self.status_label.pack(pady=10)

        # 3. Human Area
        self.human_frame = tk.Frame(self.root, bg="#2E8B57")
        self.human_frame.pack(pady=5, fill="x")

        # Human bet sits at the top of the human frame, close to the board
        self.human_bet_label = tk.Label(self.human_frame, text="", font=("Arial", 12, "bold"), bg="#2E8B57",
                                        fg="#FFD700")
        self.human_bet_label.pack(pady=2)

        self.human_label = tk.Label(self.human_frame, text="👤 You (Human)", font=("Arial", 14, "bold"), bg="#2E8B57",
                                    fg="white")
        self.human_label.pack()
        self.human_stack_label = tk.Label(self.human_frame, text="Stack: --", font=("Arial", 12), bg="#2E8B57",
                                          fg="white")
        self.human_stack_label.pack()
        self.human_cards_label = tk.Label(self.human_frame, text="[ Your Cards ]", font=("Courier", 20, "bold"),
                                          bg="white", fg="black", padx=10, pady=5)
        self.human_cards_label.pack(pady=10)

        # 4. Action Controls Area
        self.control_frame = tk.Frame(self.root, bg="#222222", pady=15)
        self.control_frame.pack(fill="x", side="bottom")

        # --- ROW 1: Sizing Shortcuts ---
        self.shortcuts_frame = tk.Frame(self.control_frame, bg="#222222")
        self.shortcuts_frame.pack(side="top", pady=5)

        ttk.Button(self.shortcuts_frame, text="1/3 Pot", command=lambda: self.set_raise_fraction(1 / 3)).pack(
            side="left", padx=5)
        ttk.Button(self.shortcuts_frame, text="1/2 Pot", command=lambda: self.set_raise_fraction(1 / 2)).pack(
            side="left", padx=5)
        ttk.Button(self.shortcuts_frame, text="2/3 Pot", command=lambda: self.set_raise_fraction(2 / 3)).pack(
            side="left", padx=5)
        ttk.Button(self.shortcuts_frame, text="Pot", command=lambda: self.set_raise_fraction(1.0)).pack(side="left",
                                                                                                        padx=5)

        # --- ROW 2: Primary Actions ---
        self.actions_frame = tk.Frame(self.control_frame, bg="#222222")
        self.actions_frame.pack(side="top", pady=5)

        self.btn_fold = ttk.Button(self.actions_frame, text="Fold", command=self.action_fold)
        self.btn_fold.pack(side="left", padx=10)

        self.btn_check_call = ttk.Button(self.actions_frame, text="Check / Call", command=self.action_check_call)
        self.btn_check_call.pack(side="left", padx=10)

        self.raise_frame = tk.Frame(self.actions_frame, bg="#222222")
        self.raise_frame.pack(side="left", padx=10)

        self.raise_var = tk.DoubleVar(value=0.0)
        self.raise_scale = ttk.Scale(self.raise_frame, variable=self.raise_var, orient="horizontal", length=150,
                                     command=self.on_scale_move)
        self.raise_scale.pack(side="left", padx=5)

        self.raise_entry = ttk.Entry(self.raise_frame, textvariable=self.raise_var, width=6, font=("Arial", 12))
        self.raise_entry.pack(side="left", padx=5)

        self.btn_raise = ttk.Button(self.raise_frame, text="Raise To", command=self.action_raise)
        self.btn_raise.pack(side="left")

        self.btn_next_hand = ttk.Button(self.actions_frame, text="Next Hand", command=self.next_hand)

    def handle_enter_key(self, event=None):
        state = self.manager.get_ui_state()
        if state and state.get('is_hand_over', False):
            self.next_hand()

    def fmt_val(self, amount, state):
        """Formats raw chip values into display strings based on user preference."""
        if state['display_in_bb']:
            return f"{amount / state['bb_amount']:g} BB"
        return f"{amount:g}"

    def on_scale_move(self, val):
        self.raise_var.set(round(float(val), 1))

    def set_raise_fraction(self, fraction):
        state = self.manager.get_ui_state()
        if not state or state['is_hand_over']: return

        current_pot = state['pot']
        amt_to_call = state['amount_to_call']

        target_raw = amt_to_call + fraction * (current_pot + amt_to_call)

        min_b = state['min_bet']
        max_b = state['max_bet']

        if target_raw < min_b: target_raw = min_b
        if target_raw > max_b: target_raw = max_b

        # Convert back to display unit for the slider box
        mult = 1.0 / state['bb_amount'] if state['display_in_bb'] else 1.0
        self.raise_var.set(round(target_raw * mult, 1))

    def update_ui(self):
        state = self.manager.get_ui_state()
        if not state: return

        self.ai_stack_label.config(text=f"Stack: {self.fmt_val(state['ai_stack'], state)}")
        self.human_stack_label.config(text=f"Stack: {self.fmt_val(state['human_stack'], state)}")
        self.pot_label.config(text=f"Pot: {self.fmt_val(state['pot'], state)}")

        # Update Street Wagers visually
        if state['ai_street_bet'] > 0 and not state['is_hand_over']:
            self.ai_bet_label.config(text=f"Bet: {self.fmt_val(state['ai_street_bet'], state)}")
        else:
            self.ai_bet_label.config(text="")

        if state['human_street_bet'] > 0 and not state['is_hand_over']:
            self.human_bet_label.config(text=f"Bet: {self.fmt_val(state['human_street_bet'], state)}")
        else:
            self.human_bet_label.config(text="")

        self.human_cards_label.config(text=format_cards(state['human_cards']))
        board_text = format_cards(state['board']) if state['board'] else "[ Preflop ]"
        self.board_cards_label.config(text=board_text)

        self.status_label.config(text=state['last_message'])

        if state['is_hand_over']:
            if state['ai_cards']:
                self.ai_cards_label.config(text=format_cards(state['ai_cards']), fg="gold")
            else:
                self.ai_cards_label.config(text="[ Mucked ]", fg="#aaaaaa")

            self.pot_label.config(text=f"Hand Over! Final Pot: {self.fmt_val(state['pot'], state)}")
            self.status_label.config(text=f"{state['last_message']}  =>  {state['winner_message']}")

            self.toggle_buttons(active=False)
            self.btn_next_hand.pack(side="right", padx=20)
            return
        else:
            self.ai_cards_label.config(text="[ Hidden ]", fg="white")

        is_my_turn = state['current_actor'] == "Human"
        self.toggle_buttons(active=is_my_turn)

        if is_my_turn:
            if state['can_check']:
                self.btn_check_call.config(text="Check")
            else:
                self.btn_check_call.config(text=f"Call ({self.fmt_val(state['amount_to_call'], state)})")

            if state['min_bet'] is not None and state['max_bet'] is not None:
                mult = 1.0 / state['bb_amount'] if state['display_in_bb'] else 1.0
                min_disp = state['min_bet'] * mult
                max_disp = state['max_bet'] * mult

                self.raise_scale.config(from_=min_disp, to=max_disp)

                current_val = self.raise_var.get()
                if current_val < min_disp or current_val > max_disp:
                    self.raise_var.set(round(min_disp, 1))

    def toggle_buttons(self, active):
        state = tk.NORMAL if active else tk.DISABLED
        self.btn_fold.state([f"!disabled" if active else "disabled"])
        self.btn_check_call.state([f"!disabled" if active else "disabled"])
        self.btn_raise.state([f"!disabled" if active else "disabled"])
        self.raise_entry.config(state=state)
        self.raise_scale.state([f"!disabled" if active else "disabled"])

        for child in self.shortcuts_frame.winfo_children():
            child.state([f"!disabled" if active else "disabled"])

    def action_fold(self):
        if self.manager.process_human_action('FOLD'): self.update_ui()

    def action_check_call(self):
        if self.manager.process_human_action('CHECK_CALL'): self.update_ui()

    def action_raise(self):
        try:
            amount_disp = float(self.raise_var.get())
            state = self.manager.get_ui_state()

            mult = 1.0 / state['bb_amount'] if state['display_in_bb'] else 1.0
            min_disp = state['min_bet'] * mult
            max_disp = state['max_bet'] * mult

            # Adding a tiny epsilon to the condition to gracefully handle floating point rounding artifacts
            if amount_disp < min_disp - 0.01 or amount_disp > max_disp + 0.01:
                messagebox.showwarning("Invalid Bet",
                                       f"Bet must be between {min_disp:g} and {max_disp:g}")
                return

            raw_amount = amount_disp / mult
            if self.manager.process_human_action('RAISE', bet_amount=raw_amount): self.update_ui()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")

    def next_hand(self):
        self.btn_next_hand.pack_forget()
        self.manager.start_new_hand()
        self.update_ui()