/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Brand palette from the design system
        lime: {
          DEFAULT: "#D5FF40",
          50: "#F7FFD9",
          100: "#EEFFB3",
          200: "#E3FF7D",
          300: "#D5FF40",
          400: "#BCE62C",
          500: "#9FC91A",
          600: "#7FA112",
          700: "#5B7409",
          800: "#3D4D05",
          900: "#232B02",
        },
        bone: {
          DEFAULT: "#C0C2B8",
          50: "#F4F5F2",
          100: "#E9EAE5",
          200: "#D6D8D0",
          300: "#C0C2B8",
          400: "#A2A59A",
          500: "#83877B",
          600: "#686C61",
          700: "#4E5149",
          800: "#35372F",
          900: "#1D1E19",
        },
        ink: {
          // Neutral graphite surfaces with a slight olive cast, not blue-slate
          DEFAULT: "#141814",
          50: "#495448",
          100: "#3C463C",
          200: "#313A31",
          300: "#252D25",
          400: "#1C231C",
          500: "#141814",
          600: "#0F130F",
          700: "#0A0D0A",
          800: "#050705",
          900: "#020302",
        },
      },
      fontFamily: {
        sans: [
          "Fira Sans",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        display: ["Fira Sans", "system-ui", "sans-serif"],
        mono: [
          "Fira Code",
          "JetBrains Mono",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      boxShadow: {
        "glow-lime": "0 0 0 1px rgba(213, 255, 64, 0.35), 0 10px 30px -12px rgba(213, 255, 64, 0.35)",
        "panel": "0 1px 0 rgba(255,255,255,0.04) inset, 0 18px 40px -24px rgba(0,0,0,0.6)",
      },
      borderRadius: {
        sm: "0px",
        DEFAULT: "0px",
        md: "0px",
        lg: "0px",
        xl: "0px",
        "2xl": "0px",
        "3xl": "0px",
        full: "0px",
      },
      backgroundImage: {
        "grid-ink": "radial-gradient(rgba(255,255,255,0.04) 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
};
