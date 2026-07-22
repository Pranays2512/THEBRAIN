import FastAPIProvider from "./FastAPIProvider";
import { AIClient } from "../lib/ai";

export const providers = {
    fastapi: new AIClient(
        new FastAPIProvider()
    )
};

export default providers;
