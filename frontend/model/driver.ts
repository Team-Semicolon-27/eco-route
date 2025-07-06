import mongoose, {model, Model, models, Schema} from "mongoose";

export interface IDriver extends Document {
    userid: mongoose.Types.ObjectId;
    manager: mongoose.Types.ObjectId;
    city: string;
    state: string;
}

export const DriverSchema = new Schema<IDriver>({
    userid: { type: mongoose.Schema.Types.ObjectId, required: true, ref: "User" },
    manager: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
    city: { type: String, required: true },
    state: { type: String, required: true },
})

export const Driver: Model<IDriver> = models.Driver || model<IDriver>("Driver", DriverSchema);