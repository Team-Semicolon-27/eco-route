import mongoose, {model, Model, models, Schema} from "mongoose";

export interface IDriver extends Document {
    userId: mongoose.Types.ObjectId;
    manager: mongoose.Types.ObjectId | null;
    country: string;
    city: string;
    state: string;
}

export const DriverSchema = new Schema<IDriver>({
    userId: { type: mongoose.Schema.Types.ObjectId, required: true, ref: "User", unique: true },
    manager: { type: mongoose.Schema.Types.ObjectId, ref: "User", default: null },
    country: { type: String, required: true },
    city: { type: String, required: true },
    state: { type: String, required: true },
})

export const Driver: Model<IDriver> = models.Driver || model<IDriver>("Driver", DriverSchema);