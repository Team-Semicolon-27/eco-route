import mongoose, {model, Model, models, Schema} from "mongoose";

export interface IManager  extends Document {
    userId: mongoose.Types.ObjectId;
    vehicles: mongoose.Types.ObjectId[];
}

export const ManagerSchema = new Schema<IManager>({
    userId: { type: mongoose.Schema.Types.ObjectId, required: true, ref: "User" },
    vehicles: [
        { type: mongoose.Schema.Types.ObjectId, ref: "Vehicle" },
    ]
})

export const Manager: Model<IManager> = models.Manager || model<IManager>("Manager", ManagerSchema);