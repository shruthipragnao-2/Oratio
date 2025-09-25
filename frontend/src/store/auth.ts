import { create } from "zustand";

interface AuthState {
    isAuthenticated: boolean;
    token: string | null;
    login: (token: string) => void;
    logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
    isAuthenticated: !!localStorage.getItem("oratio_token"),
    token: localStorage.getItem("oratio_token"),
    login: (token: string) => {
        localStorage.setItem("oratio_token", token);
        set({ isAuthenticated: true, token });
    },
    logout: () => {
        localStorage.removeItem("oratio_token");
        set({ isAuthenticated: false, token: null });
    },
}));


